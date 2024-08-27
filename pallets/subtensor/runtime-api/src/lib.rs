#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use pallet_subtensor::{
    SerializableEpochResult, SubtensorBondData, SubtensorWeightData, WeightOptimizationParams,
};

// Here we declare the runtime API. It is implemented it the `impl` block in
// src/neuron_info.rs, src/subnet_info.rs, and src/delegate_info.rs
sp_api::decl_runtime_apis! {
    pub trait DelegateInfoRuntimeApi {
        fn get_delegates() -> Vec<u8>;
        fn get_delegate( delegate_account_vec: Vec<u8> ) -> Vec<u8>;
        fn get_delegated( delegatee_account_vec: Vec<u8> ) -> Vec<u8>;
    }

    pub trait NeuronInfoRuntimeApi {
        fn get_neurons(netuid: u16) -> Vec<u8>;
        fn get_neuron(netuid: u16, uid: u16) -> Vec<u8>;
        fn get_neurons_lite(netuid: u16) -> Vec<u8>;
        fn get_neuron_lite(netuid: u16, uid: u16) -> Vec<u8>;
    }

    pub trait SubnetInfoRuntimeApi {
        fn get_subnet_info(netuid: u16) -> Vec<u8>;
        fn get_subnets_info() -> Vec<u8>;
        fn get_subnet_hyperparams(netuid: u16) -> Vec<u8>;
    }

    pub trait StakeInfoRuntimeApi {
        fn get_stake_info_for_coldkey( coldkey_account_vec: Vec<u8> ) -> Vec<u8>;
        fn get_stake_info_for_coldkeys( coldkey_account_vecs: Vec<Vec<u8>> ) -> Vec<u8>;
    }

    pub trait SubnetRegistrationRuntimeApi {
        fn get_network_registration_cost() -> u64;
    }

    pub trait SubtensorCustomApi {
        fn subtensor_epoch(netuid: u16, incentive: Option<bool>, exclude_uid: Option<u16>) -> SerializableEpochResult;
        fn subtensor_active_stake(netuid: u16, exclude_uid: Option<u16>) -> Vec<String>;
        fn subtensor_consensus(netuid: u16, exclude_uid: Option<u16>) -> Vec<String>;
        fn subtensor_bond_data(netuid: u16, exclude_uid: Option<u16>) -> SubtensorBondData;
        fn subtensor_weights(netuid: u16, exclude_uid: Option<u16>) -> SubtensorWeightData;
        fn subtensor_dividends(netuid: u16, exclude_uid: Option<u16>) -> Vec<String>;
        fn subtensor_weight_optimization(netuid: u16, exclude_uid: Option<u16>) -> WeightOptimizationParams;
        fn subtensor_simulate_emission_drain(netuid: u16) -> Vec<(String, u64)>;
        fn subtensor_simulate_emission_drain_all() -> Vec<(String,u16,u64)>;
    }
}
