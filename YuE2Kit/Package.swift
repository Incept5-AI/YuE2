// swift-tools-version: 6.1
import PackageDescription

// YuE2Kit: Swift/MLX port of the ahmadw/YuE2-3B-MLX runner (generate.py, yue2_model.py, yue2_vae.py).
// Depends only on mlx-swift, not mlx-swift-lm, so mlxchat's FoundationModelsIntegration trait trap does not apply.
let package = Package(
    name: "YuE2Kit",
    platforms: [.iOS("27.0"), .macOS("14.0")],
    products: [
        .library(name: "YuE2Kit", targets: ["YuE2Kit"]),
        .executable(name: "yue2cli", targets: ["yue2cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
    ],
    targets: [
        .target(name: "YuE2Kit", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .executableTarget(name: "yue2cli", dependencies: ["YuE2Kit"]),
    ]
)
