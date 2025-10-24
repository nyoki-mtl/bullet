use super::SparseInputType;
use std::marker::PhantomData;

/// Submitter required interface for HalfKP samples.
pub trait ShogiHalfKPSample {
    /// Returns the active feature indices for the side to move.
    fn stm_features(&self) -> &[u32];
    /// Returns the active feature indices for the opponent.
    fn ntm_features(&self) -> &[u32];
}

/// Sparse input definition for Shogi HalfKP.
#[derive(Debug)]
pub struct ShogiHalfKPInputs<S> {
    plane_count: usize,
    king_square_count: usize,
    max_active: usize,
    horizontally_mirrored: bool,
    _marker: PhantomData<S>,
}

impl<S> ShogiHalfKPInputs<S> {
    #[must_use]
    pub const fn new(
        plane_count: usize,
        king_square_count: usize,
        max_active: usize,
        horizontally_mirrored: bool,
    ) -> Self {
        Self { plane_count, king_square_count, max_active, horizontally_mirrored, _marker: PhantomData }
    }

    #[must_use]
    pub const fn plane_count(&self) -> usize {
        self.plane_count
    }

    #[must_use]
    pub const fn king_square_count(&self) -> usize {
        self.king_square_count
    }

    #[must_use]
    pub const fn horizontally_mirrored(&self) -> bool {
        self.horizontally_mirrored
    }
}

impl<S> Clone for ShogiHalfKPInputs<S> {
    fn clone(&self) -> Self {
        Self {
            plane_count: self.plane_count,
            king_square_count: self.king_square_count,
            max_active: self.max_active,
            horizontally_mirrored: self.horizontally_mirrored,
            _marker: PhantomData,
        }
    }
}

impl<S> SparseInputType for ShogiHalfKPInputs<S>
where
    S: ShogiHalfKPSample + Send + Sync + 'static,
{
    type RequiredDataType = S;

    fn num_inputs(&self) -> usize {
        self.plane_count.checked_mul(self.king_square_count).expect("HalfKP input dimension overflow")
    }

    fn max_active(&self) -> usize {
        self.max_active
    }

    fn map_features<F: FnMut(usize, usize)>(&self, sample: &Self::RequiredDataType, mut f: F) {
        let stm = sample.stm_features();
        let ntm = sample.ntm_features();
        assert_eq!(stm.len(), ntm.len(), "HalfKP feature lengths must match");
        assert!(stm.len() <= self.max_active, "HalfKP feature count exceeded max_active");

        let limit = self.plane_count.checked_mul(self.king_square_count).expect("HalfKP input dimension overflow");
        for (&s, &n) in stm.iter().zip(ntm) {
            let stm_index = usize::try_from(s).expect("HalfKP index fits usize");
            let ntm_index = usize::try_from(n).expect("HalfKP index fits usize");
            assert!(stm_index < limit && ntm_index < limit, "HalfKP index out of range");
            f(stm_index, ntm_index);
        }
    }

    fn shorthand(&self) -> String {
        if self.horizontally_mirrored {
            format!("{}x{}hm", self.plane_count, self.king_square_count)
        } else {
            format!("{}x{}", self.plane_count, self.king_square_count)
        }
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKP inputs (plane_count={}, king_squares={}, hm={})",
            self.plane_count, self.king_square_count, self.horizontally_mirrored
        )
    }
}
