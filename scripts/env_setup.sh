
sudo apt-get update && apt-get install -y libvulkan1 vulkan-tools libglvnd-dev mesa-vulkan-drivers

mkdir -p /usr/share/vulkan/icd.d \
         /usr/share/glvnd/egl_vendor.d \
         /etc/vulkan/implicit_layer.d

printf '%s\n' \
'{' \
'    "file_format_version" : "1.0.0",' \
'    "ICD": {' \
'        "library_path": "libGLX_nvidia.so.0",' \
'        "api_version" : "1.2.155"' \
'    }' \
'}' > /usr/share/vulkan/icd.d/nvidia_icd.json && \
printf '%s\n' \
'{' \
'    "file_format_version" : "1.0.0",' \
'    "ICD" : {' \
'        "library_path" : "libEGL_nvidia.so.0"' \
'    }' \
'}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
printf '%s\n' \
'{' \
'    "file_format_version" : "1.0.0",' \
'    "layer": {' \
'        "name": "VK_LAYER_NV_optimus",' \
'        "type": "INSTANCE",' \
'        "library_path": "libGLX_nvidia.so.0",' \
'        "api_version" : "1.2.155",' \
'        "implementation_version" : "1",' \
'        "description" : "NVIDIA Optimus layer",' \
'        "functions": {' \
'            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",' \
'            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"' \
'        },' \
'        "enable_environment": {' \
'            "__NV_PRIME_RENDER_OFFLOAD": "1"' \
'        },' \
'        "disable_environment": {' \
'            "DISABLE_LAYER_NV_OPTIMUS_1": ""' \
'        }' \
'    }' \
'}' > /etc/vulkan/implicit_layer.d/nvidia_layers.json

sudo apt-get install -y curl \
    apt-utils \
    tzdata \
    git wget curl vim unzip ffmpeg \
    build-essential cmake pkg-config \
    llvm meson \
    libegl1-mesa \
    libegl1-mesa-dev \
    libegl1 \
    libgl1 \
    mesa-utils \
    libsm6 libxext6 libxrender-dev \
    libosmesa6-dev \
    python3.10 \
    python3-pip

sudo sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list && sudo apt-get update && sudo apt-get build-dep -y mesa