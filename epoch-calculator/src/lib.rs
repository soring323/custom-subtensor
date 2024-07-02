use sp_std::vec::Vec;
use substrate_fixed::types::{I32F32, I64F64};

// use substrate_fixed::types::{I32F32, I64F64};

pub trait SubtensorMethods {
    type AccountId;

    fn get_emission_value(netuid: u16) -> u64;
    fn get_subnetwork_n(netuid: u16) -> u16;
    fn get_current_block_as_u64() -> u64;
    fn get_activity_cutoff(netuid: u16) -> u64;
    fn get_last_update(netuid: u16) -> Vec<u64>;
    fn get_block_at_registration(netuid: u16) -> Vec<u64>;
    fn get_total_stake_for_hotkey(hotkey: &Self::AccountId) -> u64;
    fn get_validator_permit(netuid: u16) -> Vec<bool>;
    fn get_weights_sparse(netuid: u16) -> Vec<Vec<(u16, I32F32)>>;
    fn get_float_kappa(netuid: u16) -> I32F32;
    fn get_bonds_sparse(netuid: u16) -> Vec<Vec<(u16, I32F32)>>;
    fn get_bonds_moving_average(netuid: u16) -> u64;
    fn iter_prefix_keys(netuid: u16) -> Vec<(u16, Self::AccountId)>;
}

#[allow(clippy::indexing_slicing)]
pub fn custom_epoch<T: SubtensorMethods>(netuid: u16) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    log::info!("custom_epoch(netuid: {:?})", netuid);
    let rao_emission: u64 = T::get_emission_value(netuid);
    log::info!("rao_emission: {:?}", rao_emission);
    let n: u16 = T::get_subnetwork_n(netuid);
    log::info!("n: {:?}", n);
    let current_block: u64 = T::get_current_block_as_u64();
    log::info!("current_block: {:?}", current_block);
    let activity_cutoff: u64 = T::get_activity_cutoff(netuid) as u64;
    log::info!("activity_cutoff: {:?}", activity_cutoff);
    let last_update: Vec<u64> = T::get_last_update(netuid);
    let inactive: Vec<bool> = last_update
        .iter()
        .map(|updated| *updated + activity_cutoff < current_block)
        .collect();
    let block_at_registration: Vec<u64> = T::get_block_at_registration(netuid);
    let hotkeys: Vec<(u16, T::AccountId)> = T::iter_prefix_keys(netuid);
    let mut stake_64: Vec<I64F64> = vec![I64F64::from_num(0.0); n as usize];
    for (uid_i, hotkey) in &hotkeys {
        stake_64[*uid_i as usize] = I64F64::from_num(T::get_total_stake_for_hotkey(hotkey));
    }
    inplace_normalize_64(&mut stake_64);
    let stake: Vec<I32F32> = vec_fixed64_to_fixed32(stake_64);
    let validator_permits: Vec<bool> = T::get_validator_permit(netuid);
    let validator_forbids: Vec<bool> = validator_permits.iter().map(|&b| !b).collect();
    let mut active_stake: Vec<I32F32> = stake.clone();
    inplace_mask_vector(&inactive, &mut active_stake);
    inplace_mask_vector(&validator_forbids, &mut active_stake);
    inplace_normalize(&mut active_stake);
    log::info!("active_stake: {:?}", active_stake);
    let mut weights: Vec<Vec<(u16, I32F32)>> = T::get_weights_sparse(netuid);
    weights = mask_rows_sparse(&validator_forbids, &weights);
    weights = mask_diag_sparse(&weights);
    weights = vec_mask_sparse_matrix(
        &weights,
        &last_update,
        &block_at_registration,
        &|updated, registered| updated <= registered,
    );
    inplace_row_normalize_sparse(&mut weights);
    let kappa: I32F32 = T::get_float_kappa(netuid);
    log::info!("kappa: {:?}", kappa);
    let consensus: Vec<I32F32> = weighted_median_col_sparse(&active_stake, &weights, n, kappa);
    log::info!("consensus: {:?}", consensus);
    weights = col_clip_sparse(&weights, &consensus);
    log::info!("weights: {:?}", weights);
    let validator_trust: Vec<I32F32> = row_sum_sparse(&weights);
    log::info!("validator_trust: {:?}", validator_trust);
    let mut ranks: Vec<I32F32> = matmul_sparse(&weights, &active_stake, n);
    inplace_normalize(&mut ranks); 
    let incentive: Vec<I32F32> = ranks.clone();
    log::info!("incentive: {:?}", incentive);
    let mut bonds: Vec<Vec<(u16, I32F32)>> = T::get_bonds_sparse(netuid);    
    match (|| -> Result<_, &'static str> {
        log::debug!("Starting epoch calculation");
        let mut bonds_local = &mut bonds;
        log::info!("Initial bonds done");
        *bonds_local = vec_mask_sparse_matrix(
            &bonds_local,
            &last_update,
            &block_at_registration,
            &|updated, registered| updated <= registered,
        );
        log::info!("Masked bonds done");
        inplace_col_normalize_sparse(&mut bonds_local, n);
        log::info!("Normalized bonds done");
        let mut bonds_delta: Vec<Vec<(u16, I32F32)>> = row_hadamard_sparse(&weights, &active_stake); 
        log::info!("Bonds delta done");
        inplace_col_normalize_sparse(&mut bonds_delta, n); 
        log::info!("Normalized bonds delta done");
        let bonds_moving_average: I64F64 =
            I64F64::from_num(T::get_bonds_moving_average(netuid)) / I64F64::from_num(1_000_000);
        log::debug!("Bonds moving average: {:?}", bonds_moving_average);
        let alpha: I32F32 = I32F32::from_num(1) - I32F32::from_num(bonds_moving_average);
        log::debug!("Alpha: {:?}", alpha);
        let mut ema_bonds: Vec<Vec<(u16, I32F32)>> = mat_ema_sparse(&bonds_delta, &bonds_local, alpha);
        log::info!("EMA bond done");
        inplace_col_normalize_sparse(&mut ema_bonds, n); 
        log::info!("Normalized EMA bonds: {:?}", ema_bonds);
        let mut dividends: Vec<I32F32> = matmul_transpose_sparse(&ema_bonds, &incentive);
        inplace_normalize(&mut dividends);
        log::info!("Normalized dividends: {:?}", &dividends);
        log::info!("Converting incentive, dividends, and validator_trust to f64");
        use libm::round;
        const SCALING_FACTOR: u64 = 1_000_000_000; // 1e9
        let incentive_u64: Vec<u64> = incentive
            .iter()
            .map(|x| round(x.to_num::<f64>() * SCALING_FACTOR as f64) as u64)
            .collect();
        
        let dividends_u64: Vec<u64> = dividends
            .iter()
            .map(|x| round(x.to_num::<f64>() * SCALING_FACTOR as f64) as u64)
            .collect();
        
        let validator_trust_u64: Vec<u64> = validator_trust
            .iter()
            .map(|x| round(x.to_num::<f64>() * SCALING_FACTOR as f64) as u64)
            .collect();
        Ok((incentive_u64, dividends_u64, validator_trust_u64))
    })() {
        Ok(result) => {
            log::info!("Epoch calculation completed successfully");
            result
        },
        Err(e) => {
            log::error!("Error in epoch calculation: {}", e);
            (vec![], vec![], vec![])
        }
    }
}

// Normalize an I64F64 vector in-place
fn inplace_normalize_64(vec: &mut Vec<I64F64>) {
    let sum: I64F64 = vec.iter().sum();
    if sum == I64F64::from_num(0) {
        return;
    }
    vec.iter_mut().for_each(|value| *value /= sum);
}

// Convert Vec<I64F64> to Vec<I32F32>
fn vec_fixed64_to_fixed32(vec: Vec<I64F64>) -> Vec<I32F32> {
    vec.into_iter().map(|x| I32F32::from_num(x)).collect()
}

// Apply mask to Vec<I32F32> in-place
fn inplace_mask_vector(mask: &[bool], vec: &mut Vec<I32F32>) {
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            vec[i] = I32F32::from_num(0);
        }
    }
}

// Normalize Vec<I32F32> in-place
fn inplace_normalize(vec: &mut Vec<I32F32>) {
    let sum: I32F32 = vec.iter().sum();
    if sum == I32F32::from_num(0.0) {
        return;
    }
    vec.iter_mut().for_each(|value| *value /= sum);
}

// Mask rows in a sparse matrix
fn mask_rows_sparse(mask: &[bool], matrix: &[Vec<(u16, I32F32)>]) -> Vec<Vec<(u16, I32F32)>> {
    matrix.iter().enumerate().map(|(i, row)| {
        if mask[i] {
            row.clone()
        } else {
            Vec::new()
        }
    }).collect()
}

// Mask diagonal in a sparse matrix
fn mask_diag_sparse(matrix: &[Vec<(u16, I32F32)>]) -> Vec<Vec<(u16, I32F32)>> {
    matrix.iter().enumerate().map(|(i, row)| {
        row.iter().filter_map(|&(j, value)| {
            if i as u16 == j { None } else { Some((j, value)) }
        }).collect()
    }).collect()
}

// Mask sparse matrix based on a condition
fn vec_mask_sparse_matrix(
    matrix: &[Vec<(u16, I32F32)>],
    last_update: &[u64],
    block_at_registration: &[u64],
    condition: &dyn Fn(u64, u64) -> bool,
) -> Vec<Vec<(u16, I32F32)>> {
    matrix.iter().enumerate().map(|(i, row)| {
        row.iter().filter_map(|&(j, value)| {
            if condition(last_update[i], block_at_registration[j as usize]) {
                Some((j, value))
            } else {
                None
            }
        }).collect()
    }).collect()
}

// Normalize rows of a sparse matrix in-place
fn inplace_row_normalize_sparse(matrix: &mut Vec<Vec<(u16, I32F32)>>) {
    for row in matrix.iter_mut() {
        let sum: I32F32 = row.iter().map(|&(_, value)| value).sum();
        if sum == I32F32::from_num(0.0) {
            continue;
        }
        for &mut (_, ref mut value) in row.iter_mut() {
            *value /= sum;
        }
    }
}

// Calculate weighted median of columns in a sparse matrix
fn weighted_median_col_sparse(weights: &[I32F32], matrix: &[Vec<(u16, I32F32)>], n: u16, kappa: I32F32) -> Vec<I32F32> {
    // Placeholder implementation
    vec![I32F32::from_num(0); n as usize]
}

// Clip columns of a sparse matrix
fn col_clip_sparse(matrix: &[Vec<(u16, I32F32)>], clip_values: &[I32F32]) -> Vec<Vec<(u16, I32F32)>> {
    matrix.iter().map(|row| {
        row.iter().map(|&(j, value)| {
            let clipped_value = if value > clip_values[j as usize] {
                clip_values[j as usize]
            } else {
                value
            };
            (j, clipped_value)
        }).collect()
    }).collect()
}

// Sum rows of a sparse matrix
fn row_sum_sparse(matrix: &[Vec<(u16, I32F32)>]) -> Vec<I32F32> {
    matrix.iter().map(|row| row.iter().map(|&(_, value)| value).sum()).collect()
}

// Sparse matrix-vector multiplication
fn matmul_sparse(matrix: &[Vec<(u16, I32F32)>], vec: &[I32F32], n: u16) -> Vec<I32F32> {
    (0..n).map(|i| {
        matrix[i as usize].iter().map(|&(j, value)| value * vec[j as usize]).sum()
    }).collect()
}

// Row-wise Hadamard product for sparse matrix
fn row_hadamard_sparse(matrix1: &[Vec<(u16, I32F32)>], vec: &[I32F32]) -> Vec<Vec<(u16, I32F32)>> {
    matrix1.iter().map(|row| {
        row.iter().map(|&(j, value)| (j, value * vec[j as usize])).collect()
    }).collect()
}

// Normalize columns of a sparse matrix in-place
fn inplace_col_normalize_sparse(matrix: &mut Vec<Vec<(u16, I32F32)>>, n: u16) {
    let mut col_sums = vec![I32F32::from_num(0); n as usize];
    for row in matrix.iter() {
        for &(j, value) in row.iter() {
            col_sums[j as usize] += value;
        }
    }
    for row in matrix.iter_mut() {
        for &mut (j, ref mut value) in row.iter_mut() {
            if col_sums[j as usize] != I32F32::from_num(0) {
                *value /= col_sums[j as usize];
            }
        }
    }
}

// Exponential moving average for sparse matrices
fn mat_ema_sparse(matrix1: &[Vec<(u16, I32F32)>], matrix2: &[Vec<(u16, I32F32)>], alpha: I32F32) -> Vec<Vec<(u16, I32F32)>> {
    matrix1.iter().zip(matrix2.iter()).map(|(row1, row2)| {
        row1.iter().zip(row2.iter()).map(|(&(j1, value1), &(_, value2))| {
            (j1, alpha * value1 + (I32F32::from_num(1) - alpha) * value2)
        }).collect()
    }).collect()
}

// Sparse matrix transpose multiplication
fn matmul_transpose_sparse(matrix: &[Vec<(u16, I32F32)>], vec: &[I32F32]) -> Vec<I32F32> {
    let mut result = vec![I32F32::from_num(0); matrix.len()];
    for row in matrix.iter() {
        for &(j, value) in row.iter() {
            result[j as usize] += value * vec[j as usize];
        }
    }
    result
}
