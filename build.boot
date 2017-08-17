(set-env!
  :resource-paths #{"src" "data"}
  :dependencies '[[org.clojure/clojure "1.8.0"]
                  [nightlight "1.7.0" :scope "test"]
                  [adzerk/boot-test "1.2.0" :scope "test"]
                  [org.nd4j/nd4j-native-platform "0.8.0"]
                  [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                  [org.nd4j/nd4j-api "0.8.0"]
                  [hswick/jutsu.matrix "0.0.5"]
                  [org.datavec/datavec-api "0.8.0"]]
  :repositories (conj (get-env :repositories)
                      ["clojars" {:url "https://clojars.org/repo"
                                  :username (System/getenv "CLOJARS_USER")
                                  :password (System/getenv "CLOJARS_PASS")}]))

(task-options!
  jar {:main 'jutsu.ai.core
       :manifest {"Description" "Clojure library meant to do..."}}
  pom {:version "0.0.1"
       :project 'hswick/jutsu.ai
       :description "jutsu.ai is meant to do..."
       :url "https://github.com/author/jutsu.ai"}
  push {:repo "clojars"})

(deftask deploy []
  (comp
    (pom)
    (jar)
    (push)))

;;So nightlight can still open even if there is an error in the core file
(try
  (require 'jutsu.ai.core)
  (catch Exception e (.getMessage e)))

(require
  '[nightlight.boot :refer [nightlight]]
  '[adzerk.boot-test :refer :all])

(deftask night []
  (comp
    (wait)
    (nightlight :port 4000)))

(deftask testing [] (merge-env! :source-paths #{"test"}) identity)

(deftask test-code
  []
  (comp
    (testing)
    (test)))
