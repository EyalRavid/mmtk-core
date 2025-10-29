
pub struct ChunkedStack<T> {
    chunks: Vec<Vec<T>>,
    chunk_cap: usize,
}

impl<T> ChunkedStack<T>{
    pub fn with_chunk_cap(chunk_cap: usize) -> Self {
        Self {
            chunks: vec![Vec::with_capacity(chunk_cap)],
            chunk_cap
        }
    }

    pub fn new() -> Self {
        Self::with_chunk_cap(2048)
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.chunks.len() - 1) * self.chunk_cap + self.chunks.last().unwrap().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if self.chunks.last().unwrap().len() == self.chunk_cap {
            // Start a new chunk with fixed capacity. One allocation here.
            self.chunks.push(Vec::with_capacity(self.chunk_cap));
        }
        self.chunks.last_mut().unwrap().push(value);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len() == 0 {
            return None;
        }
        // If the last chunk is empty (can happen after prior pops), drop it.
        if self.chunks.last().unwrap().is_empty(){
            self.chunks.pop();
        }
        let top_chunk = self.chunks.last_mut().unwrap();
        top_chunk.pop()
    }
    
    #[inline]
    pub fn last(&mut self) -> Option<&T> {
        if self.len() == 0 {
            return None;
        }
        if self.chunks.last().unwrap().is_empty(){
            self.chunks.pop();
        }
        self.chunks.last().unwrap().last()
    }
}