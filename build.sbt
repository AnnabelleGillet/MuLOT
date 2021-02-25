name := "MuLOT"

version := "0.1"

scalaVersion := "2.12.8"

// Spark
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1"

// Spark test
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "3.0.1_1.0.0" % "test"

// Breeze
libraryDependencies += "org.scalanlp" %% "breeze" % "1.1"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "1.1"

fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx4096M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")
parallelExecution in Test := false