/// Options for XNNPACK CPU inference.
#[derive(Clone, Debug, Default)]
pub struct XnnpackOptions {
    pub num_threads: usize,
    pub method: Option<String>,
}

impl XnnpackOptions {
    pub fn method_name(&self) -> &str {
        self.method.as_deref().unwrap_or("forward")
    }
}

/// Options for Vulkan GPU inference.
#[derive(Clone, Debug, Default)]
pub struct VulkanOptions {
    pub method: Option<String>,
}

impl VulkanOptions {
    pub fn method_name(&self) -> &str {
        self.method.as_deref().unwrap_or("forward")
    }
}
