use serde::Serialize;
use serde_json::{Map, Value};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use crate::util::ObjectReference;
use crate::util::metadata::side_metadata::spec_defs::{GRAPH_REPORT_MARK, RC_TABLE};
use crate::util::rc::RcBits;
use crate::vm::VMBinding;
use crate::vm;
use crate::Ordering;
use crate::util::address::{CLDScanPolicy, RefScanPolicy};
use crate::vm::slot::Slot;

pub type ObjectId = usize;

const GC_CYCLE_REPORT_OUTPUT_PATH: &str = "gc_cycle_report.jsonl";

#[derive(Debug, Clone, Serialize)]
pub struct HeapObjectRecord {
    pub id: ObjectId,
    pub refcount: usize,
    pub sons: Vec<ObjectId>,

    // Matches: "extra": {}
    pub extra: Map<String, Value>,
}

impl HeapObjectRecord {
    pub fn new(id: ObjectId, refcount: usize, sons: Vec<ObjectId>) -> Self {
        Self {
            id,
            refcount,
            sons,
            extra: Map::new(),
        }
    }

    pub fn with_extra(
        id: ObjectId,
        refcount: usize,
        sons: Vec<ObjectId>,
        extra: Map<String, Value>,
    ) -> Self {
        Self {
            id,
            refcount,
            sons,
            extra,
        }
    }

    pub fn add_extra_value(&mut self, key: impl Into<String>, value: Value) {
        self.extra.insert(key.into(), value);
    }

    pub fn add_extra_usize(&mut self, key: impl Into<String>, value: usize) {
        self.extra.insert(key.into(), Value::from(value));
    }

    pub fn add_extra_bool(&mut self, key: impl Into<String>, value: bool) {
        self.extra.insert(key.into(), Value::Bool(value));
    }

    pub fn add_extra_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra.insert(key.into(), Value::String(value.into()));
    }
}

#[derive(Debug, Default, Serialize)]
pub struct GcCycleReport {
    pub objects: Vec<HeapObjectRecord>,

    pub candidates: Vec<ObjectId>,
    pub roots: Vec<ObjectId>,
    pub allocated: Vec<ObjectId>,

    pub rc_freed: Vec<ObjectId>,
    pub cycle_collector_freed: Vec<ObjectId>,
}

impl GcCycleReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_object(&mut self, id: ObjectId, refcount: usize, sons: Vec<ObjectId>) {
        self.objects
            .push(HeapObjectRecord::new(id, refcount, sons));
    }

    pub fn add_object_record(&mut self, object: HeapObjectRecord) {
        self.objects.push(object);
    }

    pub fn add_candidate(&mut self, id: ObjectId) {
        self.candidates.push(id);
    }

    pub fn add_root(&mut self, id: ObjectId) {
        self.roots.push(id);
    }

    pub fn add_allocated(&mut self, id: ObjectId) {
        self.allocated.push(id);
    }

    pub fn add_rc_freed(&mut self, id: ObjectId) {
        self.rc_freed.push(id);
    }

    pub fn add_cycle_collector_freed(&mut self, id: ObjectId) {
        self.cycle_collector_freed.push(id);
    }

    pub fn clear(&mut self) {
        self.objects.clear();

        self.candidates.clear();
        self.roots.clear();
        self.allocated.clear();

        self.rc_freed.clear();
        self.cycle_collector_freed.clear();
    }

    pub fn append_to_default_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.append_to_file(GC_CYCLE_REPORT_OUTPUT_PATH)
    }

    pub fn append_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);

        serde_json::to_writer(&mut writer, self)?;
        writer.write_all(b"\n")?;
        writer.flush()?;

        Ok(())
    }

    pub fn mark_candidate_sub_graph<VM: VMBinding> (&mut self, cand: ObjectReference) {
        self.add_candidate(cand.to_raw_address().as_usize());
        let mut dfs_stack = Vec::<ObjectReference>::new();
        dfs_stack.push(cand);

        while let Some(curr) = dfs_stack.pop() {


            if GRAPH_REPORT_MARK.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 0 {
                GRAPH_REPORT_MARK.store_atomic::<u8>(curr.to_raw_address(), 1, Ordering::Relaxed);
                let mut sons: Vec::<ObjectId> = Vec::<ObjectId>::new();

                let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                    if let Some(x) = slot.load() {
                        sons.push(x.to_raw_address().as_usize());
                        dfs_stack.push(x);
                    }
                };

                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
                let id = curr.to_raw_address().as_usize();
                let refcount = (RC_TABLE.load_atomic::<RcBits>(curr.to_raw_address(), Ordering::Relaxed) - 1) as usize;
                self.add_object(id, refcount, sons);
            } 
        }
    }


    pub fn sweep_candidate_sub_graph<VM: VMBinding> (&self, cand: ObjectReference) {
        let mut dfs_stack = Vec::<ObjectReference>::new();
        dfs_stack.push(cand);
        while let Some(curr) = dfs_stack.pop() {
            if GRAPH_REPORT_MARK.load_atomic::<u8>(curr.to_raw_address(), Ordering::Relaxed) == 1 {
                GRAPH_REPORT_MARK.store_atomic::<u8>(curr.to_raw_address(), 0, Ordering::Relaxed);
                let visitor = |slot: <VM as vm::VMBinding>::VMSlot, _| {
                    if let Some(x) = slot.load() {
                        dfs_stack.push(x);
                    }
                };

                curr.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, visitor);
            } 
        }
    }
    
}