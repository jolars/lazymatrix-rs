/// How to center each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Centering {
    /// No centering.
    #[default]
    None,
    /// Subtract the column mean.
    Mean,
    /// Subtract the column minimum.
    Min,
}

/// How to scale each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling.
    #[default]
    None,
    /// Divide by the (population) standard deviation.
    Sd,
    /// Divide by the maximum absolute value.
    MaxAbs,
    /// Divide by the 1-norm.
    L1,
    /// Divide by the 2-norm.
    L2,
    /// Divide by the range, `max - min`.
    Range,
}

/// A full normalization specification: an independent [`Centering`] and
/// [`Scaling`] choice.
///
/// When both are active, scales are computed from the **centered** columns
/// Non-translation-invariant scales are computed from centered columns (see
/// [`ColumnStats::col_l1_centered`](crate::traits::ColumnStats::col_l1_centered),
/// [`ColumnStats::col_l2_centered`](crate::traits::ColumnStats::col_l2_centered),
/// and
/// [`ColumnStats::col_maxabs_centered`](crate::traits::ColumnStats::col_maxabs_centered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Normalization {
    pub center: Centering,
    pub scale: Scaling,
}

impl Normalization {
    /// Build a specification from its two axes.
    pub fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }
}
