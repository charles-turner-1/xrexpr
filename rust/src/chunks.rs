
pub enum ChunkSpec {
    SingleSize(i64),
    Auto,
    ByteSize(String),
    FullDim,
    NoChange,
    Blockseq(Vec<i64>),
    OpaqueChunk,
}
