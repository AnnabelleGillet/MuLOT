name := "MuLOT"

version := "0.3"

scalaVersion := "2.12.8"

// Spark
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1"

// Breeze
libraryDependencies += "org.scalanlp" %% "breeze" % "1.1"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "1.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5"
fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx4096M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")
parallelExecution in Test := false