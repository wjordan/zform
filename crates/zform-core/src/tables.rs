//! A file's tables: columns of values its blocks look up by key, stored once with the dictionary.
//! A block column names a table column as a peer with [`TABLE_PEER`] set.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::cells::Cells;
use crate::payload::graph::{CoreDecoder, ExtInputs, GraphNode, PayloadGraph};
use crate::wire::{Cursor, DecodeBudget, Error, Result, put_bytes, put_uvarint};

/// Set in a graph's peer id when it names table column `id & !TABLE_PEER`.
pub const TABLE_PEER: u32 = 1 << 31;

/// Table columns, each a payload graph whose peers are earlier table columns. A column is
/// decoded when a block first reads it.
pub struct Tables {
    wire: OnceLock<Vec<u8>>,
    /// A writer's tables encode their wire when it is first asked for.
    encode: Mutex<Option<Encode>>,
    graphs: Vec<PayloadGraph>,
    budget: DecodeBudget,
    cols: Vec<OnceLock<Arc<Cells>>>,
    /// By column: each distinct value's id, the rank of its first occurrence.
    ids: Vec<OnceLock<Ids>>,
}

type Encode = Box<dyn FnOnce() -> Vec<u8> + Send>;

impl std::fmt::Debug for Tables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.wire.get().map(Vec::len);
        write!(f, "Tables({} columns, {bytes:?} bytes)", self.len())
    }
}

impl Tables {
    /// Tables of `graphs`, read back as a reader reads them.
    pub fn new(graphs: &[PayloadGraph], budget: &DecodeBudget) -> Result<Self> {
        Self::decode(&Self::wire_of(graphs), budget)
    }

    /// Tables of `cols`, whose wire `encode` writes once something asks for it.
    pub fn from_cells(cols: Vec<Cells>, encode: impl FnOnce() -> Vec<u8> + Send + 'static) -> Self {
        let n = cols.len();
        Self {
            wire: OnceLock::new(),
            encode: Mutex::new(Some(Box::new(encode))),
            graphs: Vec::new(),
            budget: DecodeBudget::default(),
            cols: cols.into_iter().map(Arc::new).map(OnceLock::from).collect(),
            ids: (0..n).map(|_| OnceLock::new()).collect(),
        }
    }

    /// The wire of `graphs`, as a dictionary frame carries it.
    pub fn wire_of(graphs: &[PayloadGraph]) -> Vec<u8> {
        let mut wire = Vec::new();
        put_uvarint(&mut wire, graphs.len() as u64);
        for g in graphs {
            put_bytes(&mut wire, &g.encode());
        }
        wire
    }

    /// Reads the table section of a dictionary frame.
    pub fn decode(wire: &[u8], budget: &DecodeBudget) -> Result<Self> {
        let mut cur = Cursor::new(wire);
        let n = cur.uvar()? as usize;
        if n == 0 || n > 64 {
            return Err(Error::Format("invalid table column count".into()));
        }
        let mut graphs = Vec::with_capacity(n);
        for c in 0..n {
            let graph = PayloadGraph::decode(cur.bytes()?, budget)?;
            if graph.ext_peers.iter().any(|&p| p as usize >= c) {
                return Err(Error::Format(
                    "table column refers to a later column".into(),
                ));
            }
            if graph.ext_pool
                || graph.ext_dict.is_some()
                || graph
                    .nodes
                    .iter()
                    .any(|n| matches!(n, GraphNode::ExtAlign { .. }))
            {
                return Err(Error::Format("table column reads block parts".into()));
            }
            graphs.push(graph);
        }
        if cur.remaining() != 0 {
            return Err(Error::Format("trailing bytes after tables".into()));
        }
        Ok(Self {
            wire: OnceLock::from(wire.to_vec()),
            encode: Mutex::new(None),
            graphs,
            budget: budget.clone(),
            cols: (0..n).map(|_| OnceLock::new()).collect(),
            ids: (0..n).map(|_| OnceLock::new()).collect(),
        })
    }

    /// Column `c`'s cells, which must be in range.
    pub fn col(&self, c: usize) -> Result<&Arc<Cells>> {
        if let Some(cells) = self.cols[c].get() {
            return Ok(cells);
        }
        let graph = &self.graphs[c];
        let peers = graph
            .ext_peers
            .iter()
            .map(|&p| self.col(p as usize).cloned())
            .collect::<Result<Vec<_>>>()?;
        let ext = ExtInputs {
            peers: &peers,
            ..ExtInputs::empty()
        };
        let cells = Arc::new(CoreDecoder::decode_cells(graph, &ext, &self.budget)?);
        Ok(self.cols[c].get_or_init(|| cells))
    }

    pub fn wire(&self) -> &[u8] {
        self.wire.get_or_init(|| {
            let encode = self.encode.lock().unwrap().take();
            encode.expect("tables have a wire or its encoder")()
        })
    }

    pub fn len(&self) -> usize {
        self.cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// Column `c`'s value ids: the rank of each value's first occurrence.
    fn ids(&self, c: usize) -> Result<&Ids> {
        if let Some(ids) = self.ids[c].get() {
            return Ok(ids);
        }
        let ids = Ids::new(self.col(c)?);
        Ok(self.ids[c].get_or_init(|| ids))
    }

    /// Ids of `cells`, values of a block column, joint with table column `c`'s: a value the
    /// column holds keeps its id there, and any other value takes the next id past them.
    pub fn joint_ids<'v>(
        &self,
        c: usize,
        cells: impl IntoIterator<Item = &'v [u8]>,
    ) -> Result<Vec<u32>> {
        let (table, col) = (self.ids(c)?, self.col(c)?);
        let mut fresh: HashMap<&[u8], u32> = HashMap::new();
        Ok(cells
            .into_iter()
            .map(|v| match table.get(col, v) {
                Some(id) => id,
                None => {
                    let next = table.len + fresh.len() as u32;
                    *fresh.entry(v).or_insert(next)
                }
            })
            .collect())
    }
}

/// A column's distinct values, found by hash and confirmed against the column's cells.
struct Ids {
    /// Open addressing: each slot `0` or one more than a value's first row, with its id.
    slots: Vec<(u32, u32)>,
    len: u32,
}

impl Ids {
    fn hash(v: &[u8]) -> usize {
        crate::wire::xxh64(v, 0) as usize
    }

    fn new(col: &Cells) -> Self {
        let mask = (col.len() * 2).next_power_of_two().max(2) - 1;
        let mut ids = Self {
            slots: vec![(0, 0); mask + 1],
            len: 0,
        };
        for (row, v) in col.iter().enumerate() {
            let mut at = Self::hash(v) & mask;
            loop {
                match ids.slots[at] {
                    (0, _) => {
                        ids.slots[at] = (row as u32 + 1, ids.len);
                        ids.len += 1;
                        break;
                    }
                    (r, _) if col.get(r as usize - 1) == Some(v) => break,
                    _ => at = (at + 1) & mask,
                }
            }
        }
        ids
    }

    fn get(&self, col: &Cells, v: &[u8]) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut at = Self::hash(v) & mask;
        loop {
            match self.slots[at] {
                (0, _) => return None,
                (r, id) if col.get(r as usize - 1) == Some(v) => return Some(id),
                _ => at = (at + 1) & mask,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::Dictionary;
    use crate::packs::PackRegistry;
    use crate::payload::graph::GraphBuilder;

    fn cols(rows: &[(&str, &str)]) -> Vec<Vec<Vec<u8>>> {
        let (keys, vals) = rows
            .iter()
            .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
            .unzip();
        vec![keys, vals]
    }

    #[test]
    fn block_ids_are_joint_with_table_ids() {
        let budget = DecodeBudget::default();
        let rows = [("a", "1"), ("b", "2"), ("a", "3"), ("c", "4"), ("b", "5")];
        let t = PackRegistry::new().build_tables(cols(&rows));
        // A reader decodes what the writer looked values up in.
        let read = Tables::decode(t.wire(), &budget).unwrap();
        for t in [&t, &read] {
            assert_eq!(t.col(1).unwrap().get(2), Some(&b"3"[..]));
            let block: [&[u8]; 4] = [b"x", b"b", b"y", b"x"];
            assert_eq!(t.joint_ids(0, block).unwrap(), [3, 1, 4, 3]);
        }
    }

    #[test]
    fn wire_is_checked() {
        let budget = DecodeBudget::default();
        let t = PackRegistry::new().build_tables(cols(&[("a", "1")]));
        let mut trailing = t.wire().to_vec();
        trailing.push(0);
        assert!(Tables::decode(&trailing, &budget).is_err());
        assert!(Tables::decode(&[0], &budget).is_err());
        // A column may name only an earlier column as a peer.
        let mut gb = GraphBuilder::new(1);
        let root = gb.add_ext_peer(0);
        let mut wire = Vec::new();
        put_uvarint(&mut wire, 1);
        put_bytes(&mut wire, &gb.finish(root).encode());
        assert!(Tables::decode(&wire, &budget).is_err());

        // The dictionary's id covers its tables.
        let dict = Dictionary::new(b"content".to_vec()).with_tables(t);
        let mut stored = dict.store(3);
        let read = Dictionary::read(&stored, dict.id(), &budget).unwrap();
        assert_eq!(read.tables().unwrap().wire(), dict.tables().unwrap().wire());
        *stored.last_mut().unwrap() ^= 1;
        assert!(Dictionary::read(&stored, dict.id(), &budget).is_err());
    }
}
