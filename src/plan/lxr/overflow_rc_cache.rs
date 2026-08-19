use std::sync::Mutex;

use crate::util::ObjectReference;
use crate::util::rc::{RefCountHelper, MAX_REF_COUNT, RC_DEATH_TRANSIENT};
use crate::vm::VMBinding; // or wherever VMBinding is imported from


const MAX_RC_USIZE: usize = MAX_REF_COUNT as usize;


pub struct RefCountWithOverflow<VM: VMBinding> {
    entries: Mutex<Vec<(ObjectReference, usize)>>,
    rc: RefCountHelper<VM>,
}

impl<VM: VMBinding> RefCountWithOverflow<VM> {

    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            rc: RefCountHelper::NEW,
        }
    }

    pub fn inc(&self, o: ObjectReference) -> usize {
        // `self.rc.inc` is a CAS retry loop internally, but it returns exactly one
        // `Result` per logical inc, so matching on that result counts each operation
        // once however many times the CAS was retried.
        match self.rc.inc(o) {
            Ok(prev) => {
                // Fast: the count was below MAX_REF_COUNT and the side-metadata CAS
                // completed the operation. The overflow table was never consulted.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_fast();
                prev as usize
            }

            Err(MAX_REF_COUNT) => {
                // Slow: the count is saturated, so this inc must go through the
                // mutex-protected Vec below. Counted here, before the lock, so that the
                // hit and insert cases below share this single increment.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::inc_slow();
                let mut entries = self.entries.lock().unwrap();

                for (key, rc) in entries.iter_mut() {
                    if *key == o {
                        let prev = *rc;
                        *rc += 1;
                        return prev;
                    }
                }

                // No overflow entry yet:
                // real RC was MAX, now it becomes MAX + 1.
                entries.push((o, MAX_RC_USIZE + 1));
                MAX_RC_USIZE
            }

            Err(other) => {
                panic!("unexpected RC increment error: {:?}", other);
            }
        }
    }

    pub fn dec(&self, o: ObjectReference) -> usize {
        // As in `inc`: one `Result` per logical dec regardless of internal CAS retries.
        match self.rc.dec(o) {
            Ok(prev_rc) => {
                // Fast: the count was neither 0 nor MAX_REF_COUNT, so the side-metadata
                // CAS completed the operation without touching the overflow table.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_fast();
                prev_rc as usize
            }

            Err(MAX_REF_COUNT) => {
                // Slow: the count is saturated. Counted once here, which covers all three
                // outcomes below -- entry decremented, entry removed, or a full scan that
                // misses and falls back to `dec_unconditionally`. That fallback re-enters
                // `RefCountHelper`, not this method, so it cannot count a second time.
                #[cfg(feature = "lxr_rc_path_stats")]
                super::rc_path_stats::dec_slow();
                let mut entries = self.entries.lock().unwrap();

                for i in 0..entries.len() {
                    if entries[i].0 == o {
                        let prev_rc = entries[i].1;

                        if prev_rc == MAX_RC_USIZE + 1 {
                            // Real RC goes from MAX + 1 to MAX.
                            // The normal RC table is already saturated at MAX,
                            // so we only remove the overflow entry.
                            entries.swap_remove(i);
                        } else {
                            entries[i].1 = prev_rc - 1;
                        }

                        return prev_rc;
                    }
                }

                // Important: release the cache lock before touching RC table again.
                drop(entries);

                // No overflow entry means the real RC was exactly MAX_REF_COUNT.
                // Now decrement the table from MAX to MAX - 1.
                self.rc.dec_unconditionally(o) as usize
            }

            Err(other) => {
                panic!("unexpected RC decrement error: {:?}", other);
            }
        }
    }

    pub fn get(&self, o: ObjectReference) -> usize {
        let table_rc = self.rc.count(o) as usize;

        if table_rc != MAX_RC_USIZE {
            // Fast: a single side-metadata load answered the query. Note `get`'s boundary
            // is a plain load-and-compare, not a CAS result like `inc`/`dec`.
            #[cfg(feature = "lxr_rc_path_stats")]
            super::rc_path_stats::get_fast();
            return table_rc;
        }

        // Slow: the count is saturated, so the table must be consulted to tell
        // MAX_REF_COUNT from anything above it. Counted before the lock, so the hit and
        // the miss (which still scans the whole Vec, then returns `table_rc`) share it.
        #[cfg(feature = "lxr_rc_path_stats")]
        super::rc_path_stats::get_slow();
        let entries = self.entries.lock().unwrap();

        entries
            .iter()
            .find(|(key, _)| *key == o)
            .map(|(_, cached_rc)| *cached_rc)
            .unwrap_or(table_rc)
    }

    pub fn is_alive(&self, o: ObjectReference) -> bool {
       self.rc.count(o) as usize > RC_DEATH_TRANSIENT 
    }


    /// Number of entries currently stored in the overflow vector.
    pub fn num_entries(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Memory allocated for the overflow vector's backing storage, in bytes.
    pub fn capacity(&self) -> usize {
        let entries = self.entries.lock().unwrap();

        entries.capacity()
    } 

}