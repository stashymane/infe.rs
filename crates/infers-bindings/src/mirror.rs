/// Generates bidirectional `From` impls between a UniFFI bindings enum and its core
/// mirror, with a compile-time assertion that variant counts match.
#[macro_export]
macro_rules! uniffi_mirror {
    (
        $binding:ident <=> $core:ty,
        [$($v:ident),* $(,)?]
    ) => {
        uniffi_mirror! {
            $binding <=> $core,
            variants: [$( $v <=> $v, )*]
        }
    };

    (
        $binding:ident <=> $core:ty,
        variants: [$($bind:ident <=> $core_var:ident),* $(,)?]
    ) => {
        const _: () = {
            const CORE_COUNT: usize = std::mem::variant_count::<$core>();
            const BIND_COUNT: usize = [$(stringify!($bind),)*].len();
            const _: [(); CORE_COUNT] = [(); BIND_COUNT];
        };

        impl From<$core> for $binding {
            fn from(value: $core) -> Self {
                match value {
                    $( <$core>::$core_var => $binding::$bind, )*
                }
            }
        }

        impl From<$binding> for $core {
            fn from(value: $binding) -> Self {
                match value {
                    $( $binding::$bind => <$core>::$core_var, )*
                }
            }
        }
    };
}
