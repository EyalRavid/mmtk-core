use crate::scheduler::affinity::{get_total_num_cpus, CoreId};
use crate::util::constants::LOG_BYTES_IN_MBYTE;
use crate::util::Address;
use std::default::Default;
use std::fmt::Debug;
use std::str::FromStr;
use strum_macros::EnumString;

/// The default stress factor. This is set to the max usize,
/// which means we will never trigger a stress GC for the default value.
pub const DEFAULT_STRESS_FACTOR: usize = usize::MAX;

/// The zeroing approach to use for new object allocations.
/// Affects each plan differently.
#[derive(Copy, Clone, EnumString, Debug)]
pub enum NurseryZeroingOptions {
    /// Zeroing with normal temporal write.
    Temporal,
    /// Zeroing with cache-bypassing non-temporal write.
    Nontemporal,
    /// Zeroing with a separate zeroing thread.
    Concurrent,
    /// An adaptive approach using both non-temporal write and a concurrent zeroing thread.
    Adaptive,
}

/// Select a GC plan for MMTk.
#[derive(Copy, Clone, EnumString, Debug, PartialEq, Eq)]
pub enum PlanSelector {
    /// Allocation only without a collector. This is usually used for debugging.
    /// Similar to OpenJDK epsilon (<https://openjdk.org/jeps/318>).
    NoGC,
    /// A semi-space collector, which divides the heap into two spaces and
    /// copies the live objects into the other space for every GC.
    SemiSpace,
    /// A generational collector that uses a copying nursery, and the semi-space policy as its mature space.
    GenCopy,
    /// A generational collector that uses a copying nursery, and Immix as its mature space.
    GenImmix,
    /// A mark-sweep collector, which marks live objects and sweeps dead objects during GC.
    MarkSweep,
    /// A debugging collector that allocates memory at page granularity, and protects pages for dead objects
    /// to prevent future access.
    PageProtect,
    /// A mark-region collector that allows an opportunistic defragmentation mechanism.
    Immix,
    /// A mark-compact collector that marks objects and performs Cheney-style copying.
    MarkCompact,
    /// An Immix collector that uses a sticky mark bit to allow generational behaviors without a copying nursery.
    StickyImmix,

    MyGC //<@<@<@<@<@<@<@<@<@<@ GUY ADDED THIS
}

/// MMTk option for perf events
///
/// The format is
/// ```
/// <event> ::= <event-name> "," <pid> "," <cpu>
/// <events> ::= <event> ";" <events> | <event> | ""
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfEventOptions {
    /// A vector of perf events in tuples of (event name, PID, CPU)
    pub events: Vec<(String, i32, i32)>,
}

impl PerfEventOptions {
    fn parse_perf_events(events: &str) -> Result<Vec<(String, i32, i32)>, String> {
        events
            .split(';')
            .filter(|e| !e.is_empty())
            .map(|e| {
                let e: Vec<&str> = e.split(',').collect();
                if e.len() != 3 {
                    Err("Please supply (event name, pid, cpu)".into())
                } else {
                    let event_name = e[0].into();
                    let pid = e[1]
                        .parse()
                        .map_err(|_| String::from("Failed to parse cpu"))?;
                    let cpu = e[2]
                        .parse()
                        .map_err(|_| String::from("Failed to parse cpu"))?;
                    Ok((event_name, pid, cpu))
                }
            })
            .collect()
    }
}

impl FromStr for PerfEventOptions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PerfEventOptions::parse_perf_events(s).map(|events| PerfEventOptions { events })
    }
}

/// The default min nursery size. This does not affect the actual space we create as nursery. It is
/// only used in the GC trigger check.
#[cfg(target_pointer_width = "64")]
pub const DEFAULT_MIN_NURSERY: usize = 2 << LOG_BYTES_IN_MBYTE;
/// The default max nursery size. This does not affect the actual space we create as nursery. It is
/// only used in the GC trigger check.
#[cfg(target_pointer_width = "64")]
pub const DEFAULT_MAX_NURSERY: usize = (1 << 20) << LOG_BYTES_IN_MBYTE;

/// The default min nursery size. This does not affect the actual space we create as nursery. It is
/// only used in the GC trigger check.
#[cfg(target_pointer_width = "32")]
pub const DEFAULT_MIN_NURSERY: usize = 2 << LOG_BYTES_IN_MBYTE;
/// The default max nursery size for 32 bits.
pub const DEFAULT_MAX_NURSERY_32: usize = 32 << LOG_BYTES_IN_MBYTE;
/// The default max nursery size. This does not affect the actual space we create as nursery. It is
/// only used in the GC trigger check.
#[cfg(target_pointer_width = "32")]
pub const DEFAULT_MAX_NURSERY: usize = DEFAULT_MAX_NURSERY_32;

/// The default min nursery size proportional to the current heap size
pub const DEFAULT_PROPORTIONAL_MIN_NURSERY: f64 = 0.25;
/// The default max nursery size proportional to the current heap size
pub const DEFAULT_PROPORTIONAL_MAX_NURSERY: f64 = 1.0;

fn always_valid<T>(_: &T) -> bool {
    true
}

/// An MMTk option of a given type.
/// This type allows us to store some metadata for the option. To get the value of an option,
/// you can simply dereference it (for example, *options.threads).
#[derive(Clone)]
pub struct MMTKOption<T: Debug + Clone> {
    /// The actual value for the option
    value: T,
    /// The validator to ensure the value is valid.
    validator: fn(&T) -> bool,
    /// Can we set this option through env vars?
    from_env_var: bool,
    /// Can we set this option through command line options/API?
    from_command_line: bool,
}

impl<T: Debug + Clone> MMTKOption<T> {
    /// Create a new MMTKOption
    pub fn new(
        value: T,
        validator: fn(&T) -> bool,
        from_env_var: bool,
        from_command_line: bool,
    ) -> Self {
        // FIXME: We should enable the following check to make sure the initial value is valid.
        // However, we cannot enable it now. For options like perf events, the validator checks
        // if the perf event feature is enabled. So when the perf event features are not enabled,
        // the validator will fail whatever value we try to set (including the initial value).
        // Ideally, we conditionally compile options based on the feature. But options! macro
        // does not allow attributes in it, so we cannot conditionally compile options.
        // let is_valid = validator(&value);
        // assert!(
        //     is_valid,
        //     "Unable to create MMTKOption: initial value {:?} is invalid",
        //     value
        // );
        MMTKOption {
            value,
            validator,
            from_env_var,
            from_command_line,
        }
    }

    /// Set the option to the given value. Returns true if the value is valid, and we set the option to the value.
    pub fn set(&mut self, value: T) -> bool {
        if (self.validator)(&value) {
            self.value = value;
            return true;
        }
        false
    }
}

// Dereference an option to get its value.
impl<T: Debug + Clone> std::ops::Deref for MMTKOption<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

macro_rules! options {
    // Verify whether we can set an option through env var or command line.
    (@verify_set_from($self: expr, $key: expr, $verify_field: ident, $($name: ident),*)) => {
        match $key {
            $(stringify!($name) => { assert!($self.$name.$verify_field, "cannot set option {} (not {})", $key, stringify!($verify_field)) }),*
            _ => panic!("Invalid Options key: {}", $key)
        }
    };

    ($($(#[$outer:meta])*$name:ident: $type:ty[env_var: $env_var:expr, command_line: $command_line:expr][$validator:expr] = $default:expr),*,) => [
        options!($(#[$outer])*$($name: $type[env_var: $env_var, command_line: $command_line, mutable: $mutable][$validator] = $default),*);
    ];
    ($($(#[$outer:meta])*$name:ident: $type:ty[env_var: $env_var:expr, command_line: $command_line:expr][$validator:expr] = $default:expr),*) => [
        /// MMTk command line options.
        #[derive(Clone)]
        pub struct Options {
            $($(#[$outer])*pub $name: MMTKOption<$type>),*
        }
        impl Options {
            /// Set an option from env var
            pub fn set_from_env_var(&mut self, s: &str, val: &str) -> bool {
                options!(@verify_set_from(self, s, from_env_var, $($name),*));
                self.set_inner(s, val)
            }

            /// Set an option from command line
            pub fn set_from_command_line(&mut self, s: &str, val: &str) -> bool {
                options!(@verify_set_from(self, s, from_command_line, $($name),*));
                self.set_inner(s, val)
            }

            /// Bulk process options. Returns true if all the options are processed successfully.
            /// This method returns false if the option string is invalid, or if it includes any invalid option.
            ///
            /// Arguments:
            /// * `options`: a string that is key value pairs separated by white spaces or commas, e.g. `threads=1 stress_factor=4096`,
            ///   or `threads=1,stress_factor=4096`
            pub fn set_bulk_from_command_line(&mut self, options: &str) -> bool {
                for opt in options.replace(",", " ").split_ascii_whitespace() {
                    let kv_pair: Vec<&str> = opt.split('=').collect();
                    if kv_pair.len() != 2 {
                        return false;
                    }

                    let key = kv_pair[0];
                    let val = kv_pair[1];
                    if !self.set_from_command_line(key, val) {
                        return false;
                    }
                }

                true
            }

            /// Set an option and run its validator for its value.
            fn set_inner(&mut self, s: &str, val: &str) -> bool {
                match s {
                    // Parse the given value from str (by env vars or by calling process()) to the right type
                    $(stringify!($name) => if let Ok(typed_val) = val.parse::<$type>() {
                        let is_set = self.$name.set(typed_val);
                        if !is_set {
                            eprintln!("Warn: unable to set {}={:?}. Invalid value. Default value will be used.", s, val);
                        }
                        is_set
                    } else {
                        eprintln!("Warn: unable to set {}={:?}. Can't parse value. Default value will be used.", s, val);
                        false
                    })*
                    _ => panic!("Invalid Options key: {}", s)
                }
            }

            /// Create an `Options` instance with built-in default settings.
            fn new() -> Self {
                Options {
                    $($name: MMTKOption::new($default, $validator, $env_var, $command_line)),*
                }
            }

            /// Read options from environment variables, and apply those settings to self.
            ///
            /// If we have environment variables that start with `MMTK_` and match any option (such
            /// as `MMTK_STRESS_FACTOR`), we set the option to its value (if it is a valid value).
            pub fn read_env_var_settings(&mut self) {
                const PREFIX: &str = "MMTK_";
                for (key, val) in std::env::vars() {
                    // strip the prefix, and get the lower case string
                    if let Some(rest_of_key) = key.strip_prefix(PREFIX) {
                        let lowercase: &str = &rest_of_key.to_lowercase();
                        match lowercase {
                            $(stringify!($name) => { self.set_from_env_var(lowercase, &val); },)*
                            _ => {}
                        }
                    }
                }
            }
        }

        impl Default for Options {
            /// By default, `Options` instance is created with built-in default settings.
            fn default() -> Self {
                Self::new()
            }
        }
    ]
}

impl Options {
    /// Check if the options are set for stress GC. If either stress_factor or analysis_factor is set,
    /// we should do stress GC.
    pub fn is_stress_test_gc_enabled(&self) -> bool {
        *self.stress_factor != DEFAULT_STRESS_FACTOR
            || *self.analysis_factor != DEFAULT_STRESS_FACTOR
    }
}

#[derive(Clone, Debug, PartialEq)]
/// AffinityKind describes how to set the affinity of GC threads. Note that we currently assume
/// that each GC thread is equivalent to an OS or hardware thread.
pub enum AffinityKind {
    /// Delegate thread affinity to the OS scheduler
    OsDefault,
    /// Assign thread affinities over a list of cores in a round robin fashion. Note that if number
    /// of threads > number of cores specified, then multiple threads will be assigned the same
    /// core.
    // XXX: Maybe using a u128 bitvector with each bit representing a core is more performant?
    RoundRobin(Vec<CoreId>),
}

impl AffinityKind {
    /// Returns an AffinityKind or String containing error. Expects the list of cores to be
    /// formatted as numbers separated by commas, including ranges. There should be no spaces
    /// between the cores in the list. For example: 0,5,8-11 specifies that the cores 0,5,8,9,10,11
    /// should be used for pinning threads. Performs de-duplication of specified cores. Note that
    /// the core list is sorted as a side-effect whenever a new core is added to the set.
    fn parse_cpulist(cpulist: &str) -> Result<AffinityKind, String> {
        let mut cpuset = vec![];

        if cpulist.is_empty() {
            return Ok(AffinityKind::OsDefault);
        }

        // Split on ',' first and then split on '-' if there is a range
        for split in cpulist.split(',') {
            if !split.contains('-') {
                if !split.is_empty() {
                    if let Ok(core) = split.parse::<u16>() {
                        cpuset.push(core);
                        cpuset.sort_unstable();
                        cpuset.dedup();
                        continue;
                    }
                }
            } else {
                // Contains a range
                let range: Vec<&str> = split.split('-').collect();
                if range.len() == 2 {
                    if let Ok(start) = range[0].parse::<u16>() {
                        if let Ok(end) = range[1].parse::<u16>() {
                            if start >= end {
                                return Err(
                                    "Starting core id in range should be less than the end"
                                        .to_string(),
                                );
                            }

                            for cpu in start..=end {
                                cpuset.push(cpu);
                                cpuset.sort_unstable();
                                cpuset.dedup();
                            }

                            continue;
                        }
                    }
                }
            }

            return Err("Core ids have been incorrectly specified".to_string());
        }

        Ok(AffinityKind::RoundRobin(cpuset))
    }

    /// Return true if the affinity is either OsDefault or the cores in the list do not exceed the
    /// maximum number of cores allocated to the program. Assumes core ids on the system are
    /// 0-indexed.
    pub fn validate(&self) -> bool {
        let num_cpu = get_total_num_cpus();

        if let AffinityKind::RoundRobin(cpuset) = self {
            for cpu in cpuset {
                if cpu >= &num_cpu {
                    return false;
                }
            }
        }

        true
    }
}

impl FromStr for AffinityKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AffinityKind::parse_cpulist(s)
    }
}

#[derive(Copy, Clone, Debug)]
/// An option that provides a min/max interface to MMTk and a Bounded/Fixed interface to the
/// user/VM.
pub enum NurserySize {
    /// A Bounded nursery has different upper and lower bounds. The size only controls the upper
    /// bound. Hence, it is considered to be a "variable size" nursery.
    Bounded {
        /// The lower bound of the nursery size in bytes. Default to [`DEFAULT_MIN_NURSERY`].
        min: usize,
        /// The upper bound of the nursery size in bytes. Default to [`DEFAULT_MAX_NURSERY`].
        max: usize,
    },
    /// A bounded nursery that is porportional to the current heap size.
    ProportionalBounded {
        /// The lower bound of the nursery size as a proportion of the current heap size. Default to [`DEFAULT_PROPORTIONAL_MIN_NURSERY`].
        min: f64,
        /// The upper bound of the nursery size as a proportion of the current heap size. Default to [`DEFAULT_PROPORTIONAL_MAX_NURSERY`].
        max: f64,
    },
    /// A Fixed nursery has the same upper and lower bounds. The size controls both the upper and
    /// lower bounds. Note that this is considered less performant than a Bounded nursery since a
    /// Fixed nursery size can be too restrictive and cause more GCs.
    Fixed(usize),
}

impl NurserySize {
    /// Return true if the values are valid.
    fn validate(&self) -> bool {
        match *self {
            NurserySize::Bounded { min, max } => min <= max,
            NurserySize::ProportionalBounded { min, max } => {
                0.0f64 < min && min <= max && max <= 1.0f64
            }
            NurserySize::Fixed(_) => true,
        }
    }
}

impl FromStr for NurserySize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Invalid format".to_string());
        }

        let variant = parts[0];
        let values: Vec<&str> = parts[1].split(',').collect();

        fn default_or_parse<T: FromStr>(val: &str, default_value: T) -> Result<T, String> {
            if val == "_" {
                Ok(default_value)
            } else {
                val.parse::<T>()
                    .map_err(|_| format!("Failed to parse {:?}", std::any::type_name::<T>()))
            }
        }

        match variant {
            "Bounded" => {
                if values.len() == 2 {
                    let min = default_or_parse(values[0], DEFAULT_MIN_NURSERY)?;
                    let max = default_or_parse(values[1], DEFAULT_MAX_NURSERY)?;
                    Ok(NurserySize::Bounded { min, max })
                } else {
                    Err("Bounded requires two values".to_string())
                }
            }
            "ProportionalBounded" => {
                if values.len() == 2 {
                    let min = default_or_parse(values[0], DEFAULT_PROPORTIONAL_MIN_NURSERY)?;
                    let max = default_or_parse(values[1], DEFAULT_PROPORTIONAL_MAX_NURSERY)?;
                    Ok(NurserySize::ProportionalBounded { min, max })
                } else {
                    Err("ProportionalBounded requires two values".to_string())
                }
            }
            "Fixed" => {
                if values.len() == 1 {
                    let size = values[0]
                        .parse::<usize>()
                        .map_err(|_| "Invalid size value".to_string())?;
                    Ok(NurserySize::Fixed(size))
                } else {
                    Err("Fixed requires one value".to_string())
                }
            }
            _ => Err("Unknown variant".to_string()),
        }
    }
}

#[cfg(test)]
mod nursery_size_parsing_tests {
    use super::*;

    #[test]
    fn test_bounded() {
        // Simple case
        let result = "Bounded:1,2".parse::<NurserySize>().unwrap();
        if let NurserySize::Bounded { min, max } = result {
            assert_eq!(min, 1);
            assert_eq!(max, 2);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default min
        let result = "Bounded:_,2".parse::<NurserySize>().unwrap();
        if let NurserySize::Bounded { min, max } = result {
            assert_eq!(min, DEFAULT_MIN_NURSERY);
            assert_eq!(max, 2);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default max
        let result = "Bounded:1,_".parse::<NurserySize>().unwrap();
        if let NurserySize::Bounded { min, max } = result {
            assert_eq!(min, 1);
            assert_eq!(max, DEFAULT_MAX_NURSERY);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default both
        let result = "Bounded:_,_".parse::<NurserySize>().unwrap();
        if let NurserySize::Bounded { min, max } = result {
            assert_eq!(min, DEFAULT_MIN_NURSERY);
            assert_eq!(max, DEFAULT_MAX_NURSERY);
        } else {
            panic!("Failed: {:?}", result);
        }
    }

    #[test]
    fn test_proportional() {
        // Simple case
        let result = "ProportionalBounded:0.1,0.8"
            .parse::<NurserySize>()
            .unwrap();
        if let NurserySize::ProportionalBounded { min, max } = result {
            assert_eq!(min, 0.1);
            assert_eq!(max, 0.8);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default min
        let result = "ProportionalBounded:_,0.8".parse::<NurserySize>().unwrap();
        if let NurserySize::ProportionalBounded { min, max } = result {
            assert_eq!(min, DEFAULT_PROPORTIONAL_MIN_NURSERY);
            assert_eq!(max, 0.8);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default max
        let result = "ProportionalBounded:0.1,_".parse::<NurserySize>().unwrap();
        if let NurserySize::ProportionalBounded { min, max } = result {
            assert_eq!(min, 0.1);
            assert_eq!(max, DEFAULT_PROPORTIONAL_MAX_NURSERY);
        } else {
            panic!("Failed: {:?}", result);
        }

        // Default both
        let result = "ProportionalBounded:_,_".parse::<NurserySize>().unwrap();
        if let NurserySize::ProportionalBounded { min, max } = result {
            assert_eq!(min, DEFAULT_PROPORTIONAL_MIN_NURSERY);
            assert_eq!(max, DEFAULT_PROPORTIONAL_MAX_NURSERY);
        } else {
            panic!("Failed: {:?}", result);
        }
    }
}

/// Select a GC trigger for MMTk.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GCTriggerSelector {
    /// GC is triggered when a fix-sized heap is full. The value specifies the fixed heap size in bytes.
    FixedHeapSize(usize),
    /// GC is triggered by internal herusticis, and the heap size is varying between the two given values.
    /// The two values are the lower and the upper bound of the heap size.
    DynamicHeapSize(usize, usize),
    /// Delegate the GC triggering to the binding. This is not supported at the moment.
    Delegated,
}

impl GCTriggerSelector {
    const K: u64 = 1024;
    const M: u64 = 1024 * Self::K;
    const G: u64 = 1024 * Self::M;
    const T: u64 = 1024 * Self::G;

    /// get max heap size
    pub fn max_heap_size(&self) -> usize {
        match self {
            Self::FixedHeapSize(s) => *s,
            Self::DynamicHeapSize(_, s) => *s,
            _ => unreachable!("Cannot get max heap size"),
        }
    }

    /// Parse a size representation, which could be a number to represents bytes,
    /// or a number with the suffix K/k/M/m/G/g. Return the byte number if it can be
    /// parsed properly, otherwise return an error string.
    fn parse_size(s: &str) -> Result<usize, String> {
        let s = s.to_lowercase();
        if s.ends_with(char::is_alphabetic) {
            let num = s[0..s.len() - 1]
                .parse::<u64>()
                .map_err(|e| e.to_string())?;
            let size = if s.ends_with('k') {
                num.checked_mul(Self::K)
            } else if s.ends_with('m') {
                num.checked_mul(Self::M)
            } else if s.ends_with('g') {
                num.checked_mul(Self::G)
            } else if s.ends_with('t') {
                num.checked_mul(Self::T)
            } else {
                return Err(format!(
                    "Unknown size descriptor: {:?}",
                    &s[(s.len() - 1)..]
                ));
            };

            if let Some(size) = size {
                size.try_into()
                    .map_err(|_| format!("size overflow: {}", size))
            } else {
                Err(format!("size overflow: {}", s))
            }
        } else {
            s.parse::<usize>().map_err(|e| e.to_string())
        }
    }

    /// Return true if the gc trigger is valid
    fn validate(&self) -> bool {
        match self {
            Self::FixedHeapSize(size) => *size > 0,
            Self::DynamicHeapSize(min, max) => min <= max,
            Self::Delegated => true,
        }
    }
}

impl FromStr for GCTriggerSelector {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use regex::Regex;
        lazy_static! {
            static ref FIXED_HEAP_REGEX: Regex =
                Regex::new(r"^FixedHeapSize:(?P<size>\d+[kKmMgGtT]?)$").unwrap();
            static ref DYNAMIC_HEAP_REGEX: Regex =
                Regex::new(r"^DynamicHeapSize:(?P<min>\d+[kKmMgGtT]?),(?P<max>\d+[kKmMgGtT]?)$")
                    .unwrap();
        }

        if s.is_empty() {
            return Err("No GC trigger policy is supplied".to_string());
        }

        if let Some(captures) = FIXED_HEAP_REGEX.captures(s) {
            return Self::parse_size(&captures["size"]).map(Self::FixedHeapSize);
        } else if let Some(captures) = DYNAMIC_HEAP_REGEX.captures(s) {
            let min = Self::parse_size(&captures["min"])?;
            let max = Self::parse_size(&captures["max"])?;
            return Ok(Self::DynamicHeapSize(min, max));
        } else if s.starts_with("Delegated") {
            return Ok(Self::Delegated);
        }

        Err(format!("Failed to parse the GC trigger option: {:?}", s))
    }
}

#[cfg(test)]
mod gc_trigger_tests {
    use super::*;

    #[test]
    fn test_parse_size() {
        // correct cases
        assert_eq!(GCTriggerSelector::parse_size("0"), Ok(0));
        assert_eq!(GCTriggerSelector::parse_size("1K"), Ok(1024));
        assert_eq!(GCTriggerSelector::parse_size("1k"), Ok(1024));
        assert_eq!(GCTriggerSelector::parse_size("2M"), Ok(2 * 1024 * 1024));
        assert_eq!(GCTriggerSelector::parse_size("2m"), Ok(2 * 1024 * 1024));
        assert_eq!(
            GCTriggerSelector::parse_size("2G"),
            Ok(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            GCTriggerSelector::parse_size("2g"),
            Ok(2 * 1024 * 1024 * 1024)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            GCTriggerSelector::parse_size("2T"),
            Ok(2 * 1024 * 1024 * 1024 * 1024)
        );

        // empty
        assert_eq!(
            GCTriggerSelector::parse_size(""),
            Err("cannot parse integer from empty string".to_string())
        );

        // negative number - we dont care about actual error message
        assert!(GCTriggerSelector::parse_size("-1").is_err());

        // no number
        assert!(GCTriggerSelector::parse_size("k").is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    fn test_parse_overflow_size() {
        assert_eq!(
            GCTriggerSelector::parse_size("4G"),
            Err("size overflow: 4294967296".to_string())
        );
        assert_eq!(GCTriggerSelector::parse_size("4294967295"), Ok(4294967295));
    }

    #[test]
    fn test_parse_fixed_heap() {
        assert_eq!(
            GCTriggerSelector::from_str("FixedHeapSize:1024"),
            Ok(GCTriggerSelector::FixedHeapSize(1024))
        );
        assert_eq!(
            GCTriggerSelector::from_str("FixedHeapSize:4m"),
            Ok(GCTriggerSelector::FixedHeapSize(4 * 1024 * 1024))
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            GCTriggerSelector::from_str("FixedHeapSize:4t"),
            Ok(GCTriggerSelector::FixedHeapSize(
                4 * 1024 * 1024 * 1024 * 1024
            ))
        );

        // incorrect
        assert!(GCTriggerSelector::from_str("FixedHeapSize").is_err());
        assert!(GCTriggerSelector::from_str("FixedHeapSize:").is_err());
        assert!(GCTriggerSelector::from_str("FixedHeapSize:-1").is_err());
    }

    #[test]
    fn test_parse_dynamic_heap() {
        assert_eq!(
            GCTriggerSelector::from_str("DynamicHeapSize:1024,2048"),
            Ok(GCTriggerSelector::DynamicHeapSize(1024, 2048))
        );
        assert_eq!(
            GCTriggerSelector::from_str("DynamicHeapSize:1024,1024"),
            Ok(GCTriggerSelector::DynamicHeapSize(1024, 1024))
        );
        assert_eq!(
            GCTriggerSelector::from_str("DynamicHeapSize:1m,2m"),
            Ok(GCTriggerSelector::DynamicHeapSize(
                1024 * 1024,
                2 * 1024 * 1024
            ))
        );

        // incorrect
        assert!(GCTriggerSelector::from_str("DynamicHeapSize:1024,1024,").is_err());
    }

    #[test]
    fn test_validate() {
        assert!(GCTriggerSelector::FixedHeapSize(1024).validate());
        assert!(GCTriggerSelector::DynamicHeapSize(1024, 2048).validate());
        assert!(GCTriggerSelector::DynamicHeapSize(1024, 1024).validate());

        assert!(!GCTriggerSelector::FixedHeapSize(0).validate());
        assert!(!GCTriggerSelector::DynamicHeapSize(2048, 1024).validate());
    }
}

// Currently we allow all the options to be set by env var for the sake of convenience.
// At some point, we may disallow this and all the options can only be set by command line.
options! {
    /// The GC plan to use.
    plan:                  PlanSelector         [env_var: true, command_line: true] [always_valid] = PlanSelector::GenImmix,
    /// Number of GC worker threads.
    threads:               usize                [env_var: true, command_line: true] [|v: &usize| *v > 0]    = num_cpus::get(),
    /// Enable an optimization that only scans the part of the stack that has changed since the last GC (not supported)
    use_short_stack_scans: bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// Enable a return barrier (not supported)
    use_return_barrier:    bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// Should we eagerly finish sweeping at the start of a collection? (not supported)
    eager_complete_sweep:  bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// Should we ignore GCs requested by the user (e.g. java.lang.System.gc)?
    ignore_system_gc:      bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// The nursery size for generational plans. It can be one of Bounded, ProportionalBounded or Fixed.
    /// The nursery size can be set like 'Fixed:8192', for example,
    /// to have a Fixed nursery size of 8192 bytes, or 'ProportionalBounded:0.2,1.0' to have a nursery size
    /// between 20% and 100% of the heap size. You can omit lower bound and upper bound to use the default
    /// value for bounded nursery by using '_'. For example, 'ProportionalBounded:0.1,_' sets the min nursery
    /// to 10% of the heap size while using the default value for max nursery.
    nursery:               NurserySize          [env_var: true, command_line: true]  [|v: &NurserySize| v.validate()]
        = NurserySize::ProportionalBounded { min: DEFAULT_PROPORTIONAL_MIN_NURSERY, max: DEFAULT_PROPORTIONAL_MAX_NURSERY },
    /// Should a major GC be performed when a system GC is required?
    full_heap_system_gc:   bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// Should finalization be disabled?
    no_finalizer:          bool                 [env_var: true, command_line: true]  [always_valid] = false,
    /// Should reference type processing be disabled?
    /// If reference type processing is disabled, no weak reference processing work is scheduled,
    /// and we expect a binding to treat weak references as strong references.
    /// We disable weak reference processing by default, as we are still working on it. This will be changed to `false`
    /// once weak reference processing is implemented properly.
    no_reference_types:    bool                 [env_var: true, command_line: true]  [always_valid] = true,
    /// The zeroing approach to use for new object allocations. Affects each plan differently. (not supported)
    nursery_zeroing:       NurseryZeroingOptions[env_var: true, command_line: true]  [always_valid] = NurseryZeroingOptions::Temporal,
    /// How frequent (every X bytes) should we do a stress GC?
    stress_factor:         usize                [env_var: true, command_line: true]  [always_valid] = DEFAULT_STRESS_FACTOR,
    /// How frequent (every X bytes) should we run analysis (a STW event that collects data)
    analysis_factor:       usize                [env_var: true, command_line: true]  [always_valid] = DEFAULT_STRESS_FACTOR,
    /// Precise stress test. Trigger stress GCs exactly at X bytes if this is true. This is usually used to test the GC correctness
    /// and will significantly slow down the mutator performance. If this is false, stress GCs will only be triggered when an allocation reaches
    /// the slow path. This means we may have allocated more than X bytes or fewer than X bytes when we actually trigger a stress GC.
    /// But this should have no obvious mutator overhead, and can be used to test GC performance along with a larger stress
    /// factor (e.g. tens of metabytes).
    precise_stress:        bool                 [env_var: true, command_line: true]  [always_valid] = true,
    /// The start of vmspace.
    vm_space_start:        Address              [env_var: true, command_line: true]  [always_valid] = Address::ZERO,
    /// The size of vmspace.
    vm_space_size:         usize                [env_var: true, command_line: true] [|v: &usize| *v > 0]    = 0xdc0_0000,
    /// Perf events to measure
    /// Semicolons are used to separate events
    /// Each event is in the format of event_name,pid,cpu (see man perf_event_open for what pid and cpu mean).
    /// For example, PERF_COUNT_HW_CPU_CYCLES,0,-1 measures the CPU cycles for the current process on all the CPU cores.
    /// Measuring perf events for work packets. NOTE that be VERY CAREFUL when using this option, as this may greatly slowdown GC performance.
    // TODO: Ideally this option should only be included when the features 'perf_counter' and 'work_packet_stats' are enabled. The current macro does not allow us to do this.
    work_perf_events:       PerfEventOptions     [env_var: true, command_line: true] [|_| cfg!(all(feature = "perf_counter", feature = "work_packet_stats"))] = PerfEventOptions {events: vec![]},
    /// Measuring perf events for GC and mutators
    // TODO: Ideally this option should only be included when the features 'perf_counter' are enabled. The current macro does not allow us to do this.
    phase_perf_events:      PerfEventOptions     [env_var: true, command_line: true] [|_| cfg!(feature = "perf_counter")] = PerfEventOptions {events: vec![]},
    /// Should we exclude perf events occurring in kernel space. By default we include the kernel.
    /// Only set this option if you know the implications of excluding the kernel!
    perf_exclude_kernel:    bool                  [env_var: true, command_line: true] [|_| cfg!(feature = "perf_counter")] = false,
    /// Set how to bind affinity to the GC Workers. Default thread affinity delegates to the OS
    /// scheduler. If a list of cores are specified, cores are allocated to threads in a round-robin
    /// fashion. The core ids should match the ones reported by /proc/cpuinfo. Core ids are
    /// separated by commas and may include ranges. There should be no spaces in the core list. For
    /// example: 0,5,8-11 specifies that cores 0,5,8,9,10,11 should be used for pinning threads.
    /// Note that in the case the program has only been allocated a certain number of cores using
    /// `taskset`, the core ids in the list should be specified by their perceived index as using
    /// `taskset` will essentially re-label the core ids. For example, running the program with
    /// `MMTK_THREAD_AFFINITY="0-4" taskset -c 6-12 <program>` means that the cores 6,7,8,9,10 will
    /// be used to pin threads even though we specified the core ids "0,1,2,3,4".
    /// `MMTK_THREAD_AFFINITY="12" taskset -c 6-12 <program>` will not work, on the other hand, as
    /// there is no core with (perceived) id 12.
    // XXX: This option is currently only supported on Linux.
    thread_affinity:        AffinityKind         [env_var: true, command_line: true] [|v: &AffinityKind| v.validate()] = AffinityKind::OsDefault,
    /// Set the GC trigger. This defines the heap size and how MMTk triggers a GC.
    /// Default to a fixed heap size of 0.5x physical memory.
    gc_trigger:             GCTriggerSelector    [env_var: true, command_line: true] [|v: &GCTriggerSelector| v.validate()] = GCTriggerSelector::FixedHeapSize((crate::util::memory::get_system_total_memory() as f64 * 0.5f64) as usize),
    /// Enable transparent hugepage support for MMTk spaces via madvise (only Linux is supported)
    /// This only affects the memory for MMTk spaces.
    transparent_hugepages: bool                  [env_var: true, command_line: true]  [|v: &bool| !v || cfg!(target_os = "linux")] = false,
    /// Count live bytes for objects in each space during a GC.
    count_live_bytes_in_gc: bool                 [env_var: true, command_line: true] [always_valid] = false
}

#[cfg(test)]
mod tests {
    use super::DEFAULT_STRESS_FACTOR;
    use super::*;
    use crate::util::options::Options;
    use crate::util::test_util::{serial_test, with_cleanup};

    #[test]
    fn no_env_var() {
        serial_test(|| {
            let mut options = Options::default();
            options.read_env_var_settings();
            assert_eq!(*options.stress_factor, DEFAULT_STRESS_FACTOR);
        })
    }

    #[test]
    fn with_valid_env_var() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_STRESS_FACTOR", "4096");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(*options.stress_factor, 4096);
                },
                || {
                    std::env::remove_var("MMTK_STRESS_FACTOR");
                },
            )
        })
    }

    #[test]
    fn with_multiple_valid_env_vars() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_STRESS_FACTOR", "4096");
                    std::env::set_var("MMTK_NO_FINALIZER", "true");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(*options.stress_factor, 4096);
                    assert!(*options.no_finalizer);
                },
                || {
                    std::env::remove_var("MMTK_STRESS_FACTOR");
                    std::env::remove_var("MMTK_NO_FINALIZER");
                },
            )
        })
    }

    #[test]
    fn with_invalid_env_var_value() {
        serial_test(|| {
            with_cleanup(
                || {
                    // invalid value, we cannot parse the value, so use the default value
                    std::env::set_var("MMTK_STRESS_FACTOR", "abc");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(*options.stress_factor, DEFAULT_STRESS_FACTOR);
                },
                || {
                    std::env::remove_var("MMTK_STRESS_FACTOR");
                },
            )
        })
    }

    #[test]
    fn with_invalid_env_var_key() {
        serial_test(|| {
            with_cleanup(
                || {
                    // invalid value, we cannot parse the value, so use the default value
                    std::env::set_var("MMTK_ABC", "42");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(*options.stress_factor, DEFAULT_STRESS_FACTOR);
                },
                || {
                    std::env::remove_var("MMTK_ABC");
                },
            )
        })
    }

    #[test]
    fn ignore_env_var() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_STRESS_FACTOR", "42");

                    let options = Options::default();
                    // Not calling read_env_var_settings here.
                    assert_eq!(*options.stress_factor, DEFAULT_STRESS_FACTOR);
                },
                || {
                    std::env::remove_var("MMTK_STRESS_FACTOR");
                },
            )
        })
    }

    #[test]
    fn test_str_option_default() {
        serial_test(|| {
            let options = Options::default();
            assert_eq!(
                *options.work_perf_events,
                PerfEventOptions { events: vec![] }
            );
        })
    }

    #[test]
    #[cfg(all(feature = "perf_counter", feature = "work_packet_stats"))]
    fn test_work_perf_events_option_from_env_var() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_WORK_PERF_EVENTS", "PERF_COUNT_HW_CPU_CYCLES,0,-1");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(
                        *options.work_perf_events,
                        PerfEventOptions {
                            events: vec![("PERF_COUNT_HW_CPU_CYCLES".into(), 0, -1)]
                        }
                    );
                },
                || {
                    std::env::remove_var("MMTK_WORK_PERF_EVENTS");
                },
            )
        })
    }

    #[test]
    #[cfg(all(feature = "perf_counter", feature = "work_packet_stats"))]
    fn test_invalid_work_perf_events_option_from_env_var() {
        serial_test(|| {
            with_cleanup(
                || {
                    // The option needs to start with "hello", otherwise it is invalid.
                    std::env::set_var("MMTK_WORK_PERF_EVENTS", "PERF_COUNT_HW_CPU_CYCLES");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    // invalid value from env var, use default.
                    assert_eq!(
                        *options.work_perf_events,
                        PerfEventOptions { events: vec![] }
                    );
                },
                || {
                    std::env::remove_var("MMTK_WORK_PERF_EVENTS");
                },
            )
        })
    }

    #[test]
    #[cfg(not(feature = "perf_counter"))]
    fn test_phase_perf_events_option_without_feature() {
        serial_test(|| {
            with_cleanup(
                || {
                    // We did not enable the perf_counter feature. The option will be invalid anyway, and will be set to empty.
                    std::env::set_var("MMTK_PHASE_PERF_EVENTS", "PERF_COUNT_HW_CPU_CYCLES,0,-1");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    // invalid value from env var, use default.
                    assert_eq!(
                        *options.work_perf_events,
                        PerfEventOptions { events: vec![] }
                    );
                },
                || {
                    std::env::remove_var("MMTK_PHASE_PERF_EVENTS");
                },
            )
        })
    }

    #[test]
    fn test_thread_affinity_invalid_option() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_THREAD_AFFINITY", "0-");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    // invalid value from env var, use default.
                    assert_eq!(*options.thread_affinity, AffinityKind::OsDefault);
                },
                || {
                    std::env::remove_var("MMTK_THREAD_AFFINITY");
                },
            )
        })
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_thread_affinity_single_core() {
        serial_test(|| {
            with_cleanup(
                || {
                    std::env::set_var("MMTK_THREAD_AFFINITY", "0");

                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(
                        *options.thread_affinity,
                        AffinityKind::RoundRobin(vec![0_u16])
                    );
                },
                || {
                    std::env::remove_var("MMTK_THREAD_AFFINITY");
                },
            )
        })
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_thread_affinity_generate_core_list() {
        serial_test(|| {
            with_cleanup(
                || {
                    let mut vec = vec![0_u16];
                    let mut cpu_list = String::new();
                    let num_cpus = get_total_num_cpus();

                    cpu_list.push('0');
                    for cpu in 1..num_cpus {
                        cpu_list.push_str(format!(",{}", cpu).as_str());
                        vec.push(cpu);
                    }

                    std::env::set_var("MMTK_THREAD_AFFINITY", cpu_list);
                    let mut options = Options::default();
                    options.read_env_var_settings();
                    assert_eq!(*options.thread_affinity, AffinityKind::RoundRobin(vec));
                },
                || {
                    std::env::remove_var("MMTK_THREAD_AFFINITY");
                },
            )
        })
    }

    #[test]
    fn test_thread_affinity_single_range() {
        serial_test(|| {
            let affinity = "0-1".parse::<AffinityKind>();
            assert_eq!(affinity, Ok(AffinityKind::RoundRobin(vec![0_u16, 1_u16])));
        })
    }

    #[test]
    fn test_thread_affinity_complex_core_list() {
        serial_test(|| {
            let affinity = "0,1-2,4".parse::<AffinityKind>();
            assert_eq!(
                affinity,
                Ok(AffinityKind::RoundRobin(vec![0_u16, 1_u16, 2_u16, 4_u16]))
            );
        })
    }

    #[test]
    fn test_thread_affinity_space_in_core_list() {
        serial_test(|| {
            let affinity = "0,1-2,4, 6".parse::<AffinityKind>();
            assert_eq!(
                affinity,
                Err("Core ids have been incorrectly specified".to_string())
            );
        })
    }

    #[test]
    fn test_thread_affinity_bad_core_list() {
        serial_test(|| {
            let affinity = "0,1-2,4,".parse::<AffinityKind>();
            assert_eq!(
                affinity,
                Err("Core ids have been incorrectly specified".to_string())
            );
        })
    }

    #[test]
    fn test_thread_affinity_range_start_greater_than_end() {
        serial_test(|| {
            let affinity = "1-0".parse::<AffinityKind>();
            assert_eq!(
                affinity,
                Err("Starting core id in range should be less than the end".to_string())
            );
        })
    }

    #[test]
    fn test_thread_affinity_bad_range_option() {
        serial_test(|| {
            let affinity = "0-1-4".parse::<AffinityKind>();
            assert_eq!(
                affinity,
                Err("Core ids have been incorrectly specified".to_string())
            );
        })
    }

    #[test]
    fn test_process_valid() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.set_from_command_line("no_finalizer", "true");
            assert!(success);
            assert!(*options.no_finalizer);
        })
    }

    #[test]
    fn test_process_invalid() {
        serial_test(|| {
            let mut options = Options::default();
            let default_no_finalizer = *options.no_finalizer;
            let success = options.set_from_command_line("no_finalizer", "100");
            assert!(!success);
            assert_eq!(*options.no_finalizer, default_no_finalizer);
        })
    }

    #[test]
    fn test_process_bulk_empty() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.set_bulk_from_command_line("");
            assert!(success);
        })
    }

    #[test]
    fn test_process_bulk_valid() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.set_bulk_from_command_line("no_finalizer=true stress_factor=42");
            assert!(success);
            assert!(*options.no_finalizer);
            assert_eq!(*options.stress_factor, 42);
        })
    }

    #[test]
    fn test_process_bulk_comma_separated_valid() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.set_bulk_from_command_line("no_finalizer=true,stress_factor=42");
            assert!(success);
            assert!(*options.no_finalizer);
            assert_eq!(*options.stress_factor, 42);
        })
    }

    #[test]
    fn test_process_bulk_invalid() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.set_bulk_from_command_line("no_finalizer=true stress_factor=a");
            assert!(!success);
        })
    }

    #[test]
    fn test_set_typed_option_valid() {
        serial_test(|| {
            let mut options = Options::default();
            let success = options.no_finalizer.set(true);
            assert!(success);
            assert!(*options.no_finalizer);
        })
    }

    #[test]
    fn test_set_typed_option_invalid() {
        serial_test(|| {
            let mut options = Options::default();
            let threads = *options.threads;
            let success = options.threads.set(0);
            assert!(!success);
            assert_eq!(*options.threads, threads);
        })
    }
}
