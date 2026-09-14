use crate::interned_ir::InternedVal;

pub struct Scalar{
    value: InternedVal,
    drops_dims: bool,
    position: Option<InternedVal>
}

pub struct ForwardSlice{
    start: Option<InternedVal>,
    stop: Option<InternedVal>,
    step: Option<InternedVal>,
}

pub struct GeneralSlice{
    start: Option<InternedVal>,
    stop: Option<InternedVal>,
    step: Option<InternedVal>
}

pub struct Positions{
    positions: Vec<InternedVal>
}

pub struct Mask{
    mask: Vec<bool>
}

pub struct Label{
    label: InternedVal
}

pub struct Advanced{
    indices: Vec<InternedVal>
}

pub enum Indexer {
    Scalar,
    ForwardSlice,
    GeneralSlice,
    Positions,
    Mask,
    Label,
    Advanced
}

