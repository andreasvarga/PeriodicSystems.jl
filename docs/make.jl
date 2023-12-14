using Documenter, PeriodicSystems
DocMeta.setdocmeta!(PeriodicSystems, :DocTestSetup, :(using PeriodicSystems); recursive=true)

makedocs(warnonly = true, 
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
          "Basic connections and operations" => "psconnect.md",
          "Basic conversions" => 
          ["psconversions.md",
           "pslifting.md"],          
   #      "order_reduction.md",
          "psanalysis.md",
          "pslyap.md",
          "psric.md",
          "psstab.md",
   #      "advanced_operations.md",
   #      "model_matching.md"
         ],
     "Utilities" => [
      "pschur.md",
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
