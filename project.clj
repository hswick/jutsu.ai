(defproject hswick/jutsu.ai "0.1.1"
  :description "Clojure wrapper for deeplearning4j intended to make machine learning on the JVM simpler"
  :url "https://github.com/author/jutsu.ai"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native-platform "0.9.1"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"]
                 [org.nd4j/nd4j-api "0.9.1"]                 
                 [org.datavec/datavec-api "0.9.1"]]

  :license {:name "Eclipse Public License 1.0"
            :url "http://opensource.org/licenses/eclipse-1.0.php"}
  
  :resource-paths ["data"] 

  :profiles {:uberjar
             {:main jutsu.ai.core}

             :user
             {:dependencies
              [[nightlight "1.7.2"]
               [hswick/jutsu.matrix "0.0.14"]
               [org.clojure/tools.namespace "0.2.11"]]}})

