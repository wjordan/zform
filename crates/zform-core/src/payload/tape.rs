//! A record of the choices an encoder made between alternative codings of some values, so
//! that like values are coded the same way without building the alternatives again.
//!
//! A choice is one byte: the site that made it and what it chose. Every choice gives a valid
//! coding, so a tape replayed on values it does not suit costs bytes and nothing else. A
//! choice that more values could turn is recorded as `OPEN`: the site then chooses again,
//! and what it chooses between is built apart from the tape.
use std::cell::RefCell;

enum Mode {
    Off,
    Record(Vec<u8>),
    Replay(Vec<u8>, usize),
}

thread_local! {
    static MODE: RefCell<Mode> = const { RefCell::new(Mode::Off) };
}

/// A choice left to be made again.
pub(crate) const OPEN: u8 = 15;

/// Whether the cost that won beat the one that lost by a fifth of it or more.
pub(crate) fn clear(won: usize, lost: usize) -> bool {
    won * 5 <= lost * 4
}

/// Where a choice is made.
#[derive(Clone, Copy)]
pub(crate) enum Site {
    /// Integers as they are, or their differences.
    Ints = 1,
    /// Integers as they are, or as how recently each last appeared.
    Mtf = 2,
    /// Parsed integers by difference, or by recency.
    ExactInt = 3,
}

/// A stretch of encoding that records its choices or makes a tape's. The tape in progress
/// around it, if any, resumes when it ends.
pub(crate) struct Scope(Option<Mode>);

impl Scope {
    pub(crate) fn record() -> Self {
        Self(Some(MODE.replace(Mode::Record(Vec::new()))))
    }

    pub(crate) fn replay(tape: &[u8]) -> Self {
        Self(Some(MODE.replace(Mode::Replay(tape.to_vec(), 0))))
    }

    /// A stretch whose choices are neither recorded nor replayed.
    pub(crate) fn apart() -> Self {
        Self(Some(MODE.replace(Mode::Off)))
    }

    /// Ends the scope, giving the choices it recorded.
    pub(crate) fn end(mut self) -> Vec<u8> {
        match MODE.replace(self.0.take().unwrap_or(Mode::Off)) {
            Mode::Record(tape) => tape,
            _ => Vec::new(),
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(outer) = self.0.take() {
            MODE.set(outer);
        }
    }
}

/// The choice the tape being replayed holds for `site`. A tape that has run out, or holds
/// another site's choice here, was made on values coded another way: the rest of the scope
/// chooses for itself.
pub(crate) fn recorded(site: Site) -> Option<u8> {
    MODE.with_borrow_mut(|mode| {
        let Mode::Replay(tape, at) = mode else {
            return None;
        };
        match tape.get(*at) {
            Some(&b) if b >> 4 == site as u8 => {
                *at += 1;
                Some(b & 15)
            }
            _ => {
                *mode = Mode::Off;
                None
            }
        }
    })
}

/// Records `choice` at `site`, then the choices made inside what it chose.
pub(crate) fn note(site: Site, choice: u8, inner: &[u8]) {
    MODE.with_borrow_mut(|mode| {
        if let Mode::Record(tape) = mode {
            tape.push((site as u8) << 4 | choice);
            tape.extend_from_slice(inner);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_replay_makes_the_recorded_choices_until_the_sites_differ() {
        assert_eq!(recorded(Site::Ints), None);
        let outer = Scope::record();
        note(Site::Mtf, 1, &[]);
        // A scope inside keeps its choices to itself until they are noted.
        let inner = Scope::record();
        note(Site::Ints, 1, &[]);
        let chosen = inner.end();
        note(Site::ExactInt, 0, &chosen);
        let tape = outer.end();
        assert_eq!(tape, [0x21, 0x30, 0x11]);

        let replay = Scope::replay(&tape);
        assert_eq!(recorded(Site::Mtf), Some(1));
        note(Site::Ints, 0, &[]);
        assert_eq!(recorded(Site::ExactInt), Some(0));
        // Another site than the tape's: this and the rest are not replayed.
        assert_eq!(recorded(Site::Mtf), None);
        assert_eq!(recorded(Site::Ints), None);
        assert!(replay.end().is_empty());
        assert_eq!(recorded(Site::Ints), None);
    }
}
