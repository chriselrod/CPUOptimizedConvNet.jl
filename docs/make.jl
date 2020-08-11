using CPUOptimizedConvNet
using Documenter

makedocs(;
    modules=[CPUOptimizedConvNet],
    authors="Chris Elrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/CPUOptimizedConvNet.jl/blob/{commit}{path}#L{line}",
    sitename="CPUOptimizedConvNet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
