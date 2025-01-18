using Documenter, PeriodicSystems
using DocumenterInterLinks
DocMeta.setdocmeta!(PeriodicSystems, :DocTestSetup, :(using PeriodicSystems); recursive=true)

links = InterLinks(
    "PeriodicMatrixEquations" => "https://andreasvarga.github.io/PeriodicMatrixEquations.jl/dev/",
);                                   

makedocs(warnonly = true, 
  modules  = [PeriodicSystems],
  sitename = "PeriodicSystems.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
      "Home"   => "index.md",
      "Library" => [ 
         "Data Types and Constructors" => [
          "ps.md"
          ],
          "Basic connections and operations" => "psconnect.md",
          "Basic conversions" => 
          ["psconversions.md",
           "pslifting.md"],          
          "psanalysis.md",
          "psstab.md"
         ],
     "Index" => "makeindex.md"
  ],
  plugins=[links]
)

deploydocs(
  repo = "github.com/andreasvarga/PeriodicSystems.jl.git",
  target = "build",
  devbranch = "master"
)
