name := "MuLOT"
ThisBuild / organization := "io.github.annabellegillet"
ThisBuild / version      := "0.7"
ThisBuild / scalaVersion := "2.13.18"//"2.12.20"

// Breeze
ThisBuild / libraryDependencies += "org.scalanlp" %% "breeze" % "2.1.0"
ThisBuild / libraryDependencies += "org.scalanlp" %% "breeze-natives" % "2.1.0"

// Logging
ThisBuild / libraryDependencies += "com.outr" %% "scribe" % "3.15.0"

// ScalaTest
ThisBuild / libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19"

ThisBuild / test in assembly := {}

lazy val core = project
    .settings(
        name := "MuLOT-core"
    )

lazy val local = project
    .settings(
        name := "MuLOT-local",
        libraryDependencies +=
            "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"
    )
    .dependsOn(core)

lazy val distributed = project
    .settings(
        name := "MuLOT-distributed",
        // Spark
        libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.8" % "provided",
        libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.8" % "provided"
    )
    .dependsOn(core)
