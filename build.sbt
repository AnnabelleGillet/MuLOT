name := "MuLOT"
ThisBuild / organization := "io.github.annabellegillet"
ThisBuild / version      := "0.5"
ThisBuild / scalaVersion := "2.12.16"

// Breeze
ThisBuild / libraryDependencies += "org.scalanlp" %% "breeze" % "1.1"
ThisBuild / libraryDependencies += "org.scalanlp" %% "breeze-natives" % "1.1"

// Logging
ThisBuild / libraryDependencies += "com.outr" %% "scribe" % "3.10.4"

// ScalaTest
ThisBuild / libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5"

ThisBuild / test in assembly := {}

lazy val core = project
    .settings(
        name := "MuLOT-core"
    )

lazy val local = project
    .settings(
        name := "MuLOT-local"
    )
    .dependsOn(core)

lazy val distributed = project
    .settings(
        name := "MuLOT-distributed",
        // Spark
        libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided",
        libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided"
    )
    .dependsOn(core)
