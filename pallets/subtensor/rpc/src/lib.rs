//! RPC interface for the custom Subtensor rpc methods

use jsonrpsee::{
    core::RpcResult,
    proc_macros::rpc,
    types::{error::ErrorObject, ErrorObjectOwned},
};
use sp_blockchain::HeaderBackend;
use sp_runtime::traits::Block as BlockT;
use std::sync::Arc;

use pallet_subtensor::{
    SerializableEpochResult, SubtensorBondData, SubtensorWeightData, WeightOptimizationParams,
};
use sp_api::ProvideRuntimeApi;

pub use subtensor_custom_rpc_runtime_api::{
    DelegateInfoRuntimeApi, NeuronInfoRuntimeApi, SubnetInfoRuntimeApi,
    SubnetRegistrationRuntimeApi, SubtensorCustomApi,
};

use sp_io::offchain;
use sp_runtime::offchain::storage::StorageValueRef;
#[rpc(client, server)]
pub trait SubtensorCustomApi<BlockHash> {
    #[method(name = "delegateInfo_getDelegates")]
    fn get_delegates(&self, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "delegateInfo_getDelegate")]
    fn get_delegate(
        &self,
        delegate_account_vec: Vec<u8>,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<u8>>;
    #[method(name = "delegateInfo_getDelegated")]
    fn get_delegated(
        &self,
        delegatee_account_vec: Vec<u8>,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<u8>>;

    #[method(name = "neuronInfo_getNeuronsLite")]
    fn get_neurons_lite(&self, netuid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "neuronInfo_getNeuronLite")]
    fn get_neuron_lite(&self, netuid: u16, uid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "neuronInfo_getNeurons")]
    fn get_neurons(&self, netuid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "neuronInfo_getNeuron")]
    fn get_neuron(&self, netuid: u16, uid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;

    #[method(name = "subnetInfo_getSubnetInfo")]
    fn get_subnet_info(&self, netuid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "subnetInfo_getSubnetsInfo")]
    fn get_subnets_info(&self, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;
    #[method(name = "subnetInfo_getSubnetHyperparams")]
    fn get_subnet_hyperparams(&self, netuid: u16, at: Option<BlockHash>) -> RpcResult<Vec<u8>>;

    #[method(name = "subnetInfo_getLockCost")]
    fn get_network_lock_cost(&self, at: Option<BlockHash>) -> RpcResult<u64>;

    //Add the new method here
    #[method(name = "subtensor_epoch")]
    fn subtensor_epoch(
        &self,
        netuid: u16,
        incentive: Option<bool>,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<SerializableEpochResult>;

    #[method(name = "subtensor_active_stake")]
    fn subtensor_active_stake(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<String>>;

    #[method(name = "subtensor_consensus")]
    fn subtensor_consensus(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<String>>;

    #[method(name = "subtensor_bond_data")]
    fn subtensor_bond_data(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<SubtensorBondData>;

    #[method(name = "subtensor_weights")]
    fn subtensor_weights(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<SubtensorWeightData>;

    #[method(name = "subtensor_dividends")]
    fn subtensor_dividends(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<String>>;

    #[method(name = "subtensor_weight_optimization")]
    fn subtensor_weight_optimization(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<BlockHash>,
    ) -> RpcResult<WeightOptimizationParams>;

    #[method(name = "subtensor_simulate_emission_drain")]
    fn subtensor_simulate_emission_drain(
        &self,
        netuid: u16,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<(String, u64)>>;

    #[method(name = "subtensor_simulate_emission_drain_all")]
    fn subtensor_simulate_emission_drain_all(
        &self,
        at: Option<BlockHash>,
    ) -> RpcResult<Vec<(String, u16, u64)>>;

    // #[method(name = "subtensor_emission_values")]
    // fn subtensor_emission_values(&self, at: Option<BlockHash>) -> RpcResult<Vec<(String, u64)>>;
}

#[rpc(client, server)]
pub trait SubtensorOffchainCustomApi<BlockHash> {
    #[method(name = "subtensor_emission_values")]
    fn subtensor_emission_values(&self, netuids: Vec<u16>) -> RpcResult<Vec<(String, u64)>>;
}

pub struct SubtensorCustom<C, P> {
    /// Shared reference to the client.
    client: Arc<C>,
    _marker: std::marker::PhantomData<P>,
}

impl<C, P> SubtensorCustom<C, P> {
    /// Creates a new instance of the TransactionPayment Rpc helper.
    pub fn new(client: Arc<C>) -> Self {
        Self {
            client,
            _marker: Default::default(),
        }
    }
}

/// Error type of this RPC api.
pub enum Error {
    /// The call to runtime failed.
    RuntimeError(String),
}

impl From<Error> for ErrorObjectOwned {
    fn from(e: Error) -> Self {
        match e {
            Error::RuntimeError(e) => ErrorObject::owned(1, e, None::<()>),
        }
    }
}

impl From<Error> for i32 {
    fn from(e: Error) -> i32 {
        match e {
            Error::RuntimeError(_) => 1,
        }
    }
}

impl<C, Block> SubtensorCustomApiServer<<Block as BlockT>::Hash> for SubtensorCustom<C, Block>
where
    Block: BlockT,
    C: ProvideRuntimeApi<Block> + HeaderBackend<Block> + Send + Sync + 'static,
    C::Api: DelegateInfoRuntimeApi<Block>,
    C::Api: NeuronInfoRuntimeApi<Block>,
    C::Api: SubnetInfoRuntimeApi<Block>,
    C::Api: SubnetRegistrationRuntimeApi<Block>,
    C::Api: SubtensorCustomApi<Block>,
{
    fn get_delegates(&self, at: Option<<Block as BlockT>::Hash>) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_delegates(at).map_err(|e| {
            Error::RuntimeError(format!("Unable to get delegates info: {:?}", e)).into()
        })
    }

    fn get_delegate(
        &self,
        delegate_account_vec: Vec<u8>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_delegate(at, delegate_account_vec).map_err(|e| {
            Error::RuntimeError(format!("Unable to get delegates info: {:?}", e)).into()
        })
    }

    fn get_delegated(
        &self,
        delegatee_account_vec: Vec<u8>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_delegated(at, delegatee_account_vec).map_err(|e| {
            Error::RuntimeError(format!("Unable to get delegates info: {:?}", e)).into()
        })
    }

    fn get_neurons_lite(
        &self,
        netuid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_neurons_lite(at, netuid).map_err(|e| {
            Error::RuntimeError(format!("Unable to get neurons lite info: {:?}", e)).into()
        })
    }

    fn get_neuron_lite(
        &self,
        netuid: u16,
        uid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_neuron_lite(at, netuid, uid).map_err(|e| {
            Error::RuntimeError(format!("Unable to get neurons lite info: {:?}", e)).into()
        })
    }

    fn get_neurons(&self, netuid: u16, at: Option<<Block as BlockT>::Hash>) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_neurons(at, netuid)
            .map_err(|e| Error::RuntimeError(format!("Unable to get neurons info: {:?}", e)).into())
    }

    fn get_neuron(
        &self,
        netuid: u16,
        uid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_neuron(at, netuid, uid)
            .map_err(|e| Error::RuntimeError(format!("Unable to get neuron info: {:?}", e)).into())
    }

    fn get_subnet_info(
        &self,
        netuid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_subnet_info(at, netuid)
            .map_err(|e| Error::RuntimeError(format!("Unable to get subnet info: {:?}", e)).into())
    }

    fn get_subnet_hyperparams(
        &self,
        netuid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_subnet_hyperparams(at, netuid)
            .map_err(|e| Error::RuntimeError(format!("Unable to get subnet info: {:?}", e)).into())
    }

    fn get_subnets_info(&self, at: Option<<Block as BlockT>::Hash>) -> RpcResult<Vec<u8>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_subnets_info(at)
            .map_err(|e| Error::RuntimeError(format!("Unable to get subnets info: {:?}", e)).into())
    }

    fn get_network_lock_cost(&self, at: Option<<Block as BlockT>::Hash>) -> RpcResult<u64> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);

        api.get_network_registration_cost(at).map_err(|e| {
            Error::RuntimeError(format!("Unable to get subnet lock cost: {:?}", e)).into()
        })
    }

    // Custom RPC methods
    fn subtensor_epoch(
        &self,
        netuid: u16,
        incentive: Option<bool>,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<SerializableEpochResult> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_epoch(at, netuid, incentive, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!("Unable to get subnet epoch values: {:?}", e)).into()
            })
    }

    fn subtensor_active_stake(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<String>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_active_stake(at, netuid, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!("Unable to get subnet active stake values: {:?}", e))
                    .into()
            })
    }

    fn subtensor_consensus(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<String>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_consensus(at, netuid, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!("Unable to get subnet consensus values: {:?}", e))
                    .into()
            })
    }

    fn subtensor_bond_data(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<SubtensorBondData> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_bond_data(at, netuid, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!("Unable to get subnet bond data: {:?}", e)).into()
            })
    }

    fn subtensor_weights(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<SubtensorWeightData> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_weights(at, netuid, exclude_uid).map_err(|e| {
            Error::RuntimeError(format!("Unable to get subnet bond data: {:?}", e)).into()
        })
    }

    fn subtensor_dividends(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<String>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_dividends(at, netuid, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!("Unable to get subnet dividends: {:?}", e)).into()
            })
    }

    fn subtensor_weight_optimization(
        &self,
        netuid: u16,
        exclude_uid: Option<u16>,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<WeightOptimizationParams> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_weight_optimization(at, netuid, exclude_uid)
            .map_err(|e| {
                Error::RuntimeError(format!(
                    "Unable to get subnet weight optimization params: {:?}",
                    e
                ))
                .into()
            })
    }

    fn subtensor_simulate_emission_drain(
        &self,
        netuid: u16,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<(String, u64)>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_simulate_emission_drain(at, netuid)
            .map_err(|e| {
                Error::RuntimeError(format!(
                    "Unable to get subnet simulate emission drain: {:?}",
                    e
                ))
                .into()
            })
    }

    fn subtensor_simulate_emission_drain_all(
        &self,
        at: Option<<Block as BlockT>::Hash>,
    ) -> RpcResult<Vec<(String, u16, u64)>> {
        let api = self.client.runtime_api();
        let at = at.unwrap_or_else(|| self.client.info().best_hash);
        api.subtensor_simulate_emission_drain_all(at).map_err(|e| {
            Error::RuntimeError(format!(
                "Unable to get subnet simulate emission drain all: {:?}",
                e
            ))
            .into()
        })
    }
}
