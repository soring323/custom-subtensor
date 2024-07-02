use sp_runtime::DispatchError;

sp_api::decl_runtime_apis! {
    pub trait SubtensorApi {
        fn trigger_custom_epoch(netuid: u16) -> Result<(), DispatchError>;
        fn get_epoch_results(netuid: u16) -> (Vec<u64>, Vec<u64>, Vec<u64>);
    }
}