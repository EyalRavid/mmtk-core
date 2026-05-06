use std::mem;
use std::sync::Mutex;

/// A thread-safe buffer pool for multi-producer, single-consumer workflows.
///
/// Multiple threads obtain [`LocalBuffer`] handles to push elements concurrently.
/// Full buffers are published to `completed`; partially-filled buffers are returned
/// to `available` for allocation reuse. Once all producers are done,
/// [`into_final_buffers`](BufferPool::into_final_buffers) merges both pools into
/// a single [`FinalBuffers`] for sequential processing.
pub struct BufferPool<T> {
    buffer_capacity: usize,
    /// Buffers that reached `buffer_capacity` and were flushed by a producer.
    completed: Mutex<Vec<Vec<T>>>,
    /// Buffers returned by finished producers. May contain residual elements
    /// that didn't fill a complete buffer. Merged into `completed` by
    /// `into_final_buffers`.
    available: Mutex<Vec<Vec<T>>>,
}

/// Thread-local handle for pushing elements into a [`BufferPool`].
///
/// Automatically returns its buffer to the pool on drop.
pub struct LocalBuffer<'a, T> {
    pool: &'a BufferPool<T>,
    current: Vec<T>,
}

/// Owned storage returned by [`BufferPool::into_final_buffers`] after all
/// producers are done. Provides iteration and cursor-based mutable access.
pub struct FinalBuffers<T> {
    buffers: Vec<Vec<T>>,
}

/// Mutable iterator over [`FinalBuffers`] that supports element removal during iteration.
pub struct FinalIterMut<'a, T> {
    buffers: &'a mut Vec<Vec<T>>,
    buf_idx: usize,
    elem_idx: usize,
    removed_current: bool,
}

impl<T> BufferPool<T> {
    pub fn new(buffer_capacity: usize) -> Self {
        debug_assert!(buffer_capacity > 0);

        Self {
            buffer_capacity,
            completed: Mutex::new(Vec::new()),
            available: Mutex::new(Vec::new()),
        }
    }

    pub fn local_buffer(&self) -> LocalBuffer<'_, T> {
        LocalBuffer {
            pool: self,
            current: self.acquire_buffer(),
        }
    }

    /// Drain both `completed` and `available` pools into a single [`FinalBuffers`].
    ///
    /// Must be called only after all [`LocalBuffer`] handles have been dropped,
    /// so that no producer can still hold a reference to this pool.
    /// Uses `get_mut` (no locking) since `&mut self` guarantees exclusive access.
    pub fn into_final_buffers(&mut self) -> FinalBuffers<T> {
        let completed = self.completed.get_mut().unwrap();
        let available = self.available.get_mut().unwrap();

        completed.append(available);

        FinalBuffers {
            buffers: std::mem::take(completed),
        }
    }

    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }

    /// Reuse a buffer from `available`, or allocate a new one.
    fn acquire_buffer(&self) -> Vec<T> {
        let mut available = self.available.lock().unwrap();
        available
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_capacity))
    }

    /// Move a full buffer into `completed`.
    fn publish_full_buffer(&self, buf: Vec<T>) {
        debug_assert_eq!(buf.len(), self.buffer_capacity);
        let mut completed = self.completed.lock().unwrap();
        completed.push(buf);
    }

    /// Return a partially-filled buffer to `available`.
    /// Empty buffers are dropped. Non-empty ones will be collected by
    /// `into_final_buffers`.
    fn return_buffer(&self, buf: Vec<T>) {
        if buf.is_empty() {
            return;
        }

        let mut available = self.available.lock().unwrap();
        available.push(buf);
    }
}

impl<T> LocalBuffer<'_, T> {
    pub fn push(&mut self, value: T) {
        self.current.push(value);

        if self.current.len() == self.pool.buffer_capacity {
            self.flush();
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            self.push(item);
        }
    }

    /// Publish the current buffer as full and acquire a fresh one.
    fn flush(&mut self) {
        debug_assert_eq!(self.current.len(), self.pool.buffer_capacity);

        let replacement = self.pool.acquire_buffer();
        let full_buf = mem::replace(&mut self.current, replacement);
        self.pool.publish_full_buffer(full_buf);
    }
}

impl<T> Drop for LocalBuffer<'_, T> {
    fn drop(&mut self) {
        let buf = mem::take(&mut self.current);
        self.pool.return_buffer(buf);
    }
}

impl<T> FinalBuffers<T> {
    pub fn into_vecs(self) -> Vec<Vec<T>> {
        self.buffers
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.buffers.into_iter().flatten()
    }

    pub fn for_each(self, mut f: impl FnMut(T)) {
        for elem in self.into_iter() {
            f(elem);
        }
    }

    pub fn iter_mut(&mut self) -> FinalIterMut<'_, T> {
        FinalIterMut {
            buffers: &mut self.buffers,
            buf_idx: 0,
            elem_idx: 0,
            // Start as true so the first next() doesn't skip element 0.
            removed_current: true,
        }
    }

    pub fn len(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    pub fn len_buffers(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.iter().all(Vec::is_empty)
    }
}

impl<'a, T> FinalIterMut<'a, T> {
    /// Return a mutable reference to the next element, or `None` if iteration is complete.
    pub fn next(&mut self) -> Option<&mut T> {
        // Advance past the previous element: if it was removed, stay in place
        // (the swapped-in element is now at the current position); otherwise step forward.
        if !self.removed_current {
            self.elem_idx += 1;
        }
        self.removed_current = false;

        // Skip empty buffers.
        while self.buf_idx < self.buffers.len() && self.elem_idx >= self.buffers[self.buf_idx].len()
        {
            self.buf_idx += 1;
            self.elem_idx = 0;
        }

        if self.buf_idx >= self.buffers.len() {
            return None;
        }

        Some(&mut self.buffers[self.buf_idx][self.elem_idx])
    }

    /// Remove the current element via swap-remove.
    ///
    /// The next call to [`next`](Self::next) will visit the element that was
    /// swapped into this position (if any), so no elements are skipped.
    pub fn swap_remove_current(&mut self) -> T {
        self.removed_current = true;
        let removed = self.buffers[self.buf_idx].swap_remove(self.elem_idx);

        if self.buffers[self.buf_idx].is_empty() {
            self.buffers.swap_remove(self.buf_idx);
        }

        removed
    }
}






#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn single_thread_collects_all_elements() {
        let mut pool = BufferPool::new(4);

        {
            let mut local = pool.local_buffer();
            for i in 0..10 {
                local.push(i);
            }
        }

        let final_buffers = pool.into_final_buffers();
        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();

        assert_eq!(values, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn multi_thread_collects_all_elements() {
        let mut pool = BufferPool::new(8);

        thread::scope(|s| {
            for tid in 0..4 {
                let pool_ref = &pool;
                s.spawn(move || {
                    let mut local = pool_ref.local_buffer();
                    for i in 0..25 {
                        local.push((tid, i));
                    }
                });
            }
        });

        let final_buffers = pool.into_final_buffers();
        let values: Vec<_> = final_buffers.into_iter().collect();

        assert_eq!(values.len(), 100);

        let mut sorted = values;
        sorted.sort();
        let expected: Vec<_> = (0..4)
            .flat_map(|tid| (0..25).map(move |i| (tid, i)))
            .collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn exact_full_buffers_work() {
        let mut pool = BufferPool::new(4);

        {
            let mut local = pool.local_buffer();
            for i in 0..8 {
                local.push(i);
            }
        }

        let final_buffers = pool.into_final_buffers();
        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();

        assert_eq!(values, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn into_final_buffers_empties_pool() {
        let mut pool = BufferPool::new(4);

        {
            let mut local = pool.local_buffer();
            for i in 0..6 {
                local.push(i);
            }
        }

        let final_buffers = pool.into_final_buffers();
        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();
        assert_eq!(values, (0..6).collect::<Vec<_>>());

        {
            let mut local = pool.local_buffer();
            for i in 10..13 {
                local.push(i);
            }
        }

        let final_buffers = pool.into_final_buffers();
        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();
        assert_eq!(values, vec![10, 11, 12]);
    }

    #[test]
    fn iter_mut_can_modify_without_removing() {
        let mut pool = BufferPool::new(3);

        {
            let mut local = pool.local_buffer();
            for i in 0..7 {
                local.push(i);
            }
        }

        let mut final_buffers = pool.into_final_buffers();
        let mut it = final_buffers.iter_mut();

        while let Some(elem) = it.next() {
            *elem *= 2;
        }

        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();

        assert_eq!(values, vec![0, 2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn iter_mut_can_swap_remove_while_iterating() {
        let mut pool = BufferPool::new(3);

        {
            let mut local = pool.local_buffer();
            for i in 0..10 {
                local.push(i);
            }
        }

        let mut final_buffers = pool.into_final_buffers();
        let mut it = final_buffers.iter_mut();

        while let Some(elem) = it.next() {
            if *elem % 2 == 1 {
                let removed = it.swap_remove_current();
                assert_eq!(removed % 2, 1);
            }
        }

        let mut values: Vec<_> = final_buffers.into_iter().collect();
        values.sort();

        assert_eq!(values, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn iter_mut_can_remove_everything() {
        let mut pool = BufferPool::new(2);

        {
            let mut local = pool.local_buffer();
            for i in 0..6 {
                local.push(i);
            }
        }

        let mut final_buffers = pool.into_final_buffers();
        let mut it = final_buffers.iter_mut();

        while let Some(_) = it.next() {
            it.swap_remove_current();
        }

        assert!(final_buffers.is_empty());
        assert_eq!(final_buffers.len_buffers(), 0);

        let values: Vec<_> = final_buffers.into_iter().collect();
        assert!(values.is_empty());
    }

    #[test]
    fn for_each_visits_all_elements_once() {
        let mut pool = BufferPool::new(4);

        {
            let mut local = pool.local_buffer();
            for i in 1..=5 {
                local.push(i);
            }
        }

        let final_buffers = pool.into_final_buffers();

        let mut sum = 0;
        final_buffers.for_each(|x| sum += x);

        assert_eq!(sum, 15);
    }

    #[test]
    fn into_vecs_exposes_underlying_buffers() {
        let mut pool = BufferPool::new(4);

        {
            let mut local = pool.local_buffer();
            for i in 0..10 {
                local.push(i);
            }
        }

        let buffers = pool.into_final_buffers().into_vecs();
        let total_len: usize = buffers.iter().map(Vec::len).sum();

        assert_eq!(total_len, 10);
        assert!(buffers.iter().all(|b| !b.is_empty()));
    }
}