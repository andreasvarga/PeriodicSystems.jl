using Documenter, PeriodicSystems
DocMeta.setdocmeta!(PeriodicSystems, :DocTestSetup, :(using PeriodicSystems); recursive=true)

makedocs(
  modules  = [PeriodicSystems],
  sitename = "PeriodicSystems.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
      "Home"   => "index.md",
      "Library" => [ 
         "Data Types and Constructors" => [
          "pstypes.md",
          "ps.md"
          ],
          "Basic conversions" => 
          ["psconversions.md",
           "pslifting.md"],
   #      "operations.md",
   #      "operations_rtf.md",
   #      "conversions.md",
   #      "order_reduction.md",
   #      "analysis.md",
   #      "factorizations.md",
   #      "advanced_operations.md",
   #      "model_matching.md"
         ],
     "Utilities" => [
      "pstools.md",
      "psconv.md",
      "slicot.md"
      ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(
  repo = "github.com/andreasvarga/PeriodicSystems.jl.git",
  target = "build",
  devbranch = "master"
)
