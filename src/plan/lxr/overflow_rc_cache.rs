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
        match self.rc.inc(o) {
            Ok(prev) => prev as usize,

            Err(MAX_REF_COUNT) => {
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
        match self.rc.dec(o) {
            Ok(prev_rc) => prev_rc as usize,

            Err(MAX_REF_COUNT) => {
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
            return table_rc;
        }

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