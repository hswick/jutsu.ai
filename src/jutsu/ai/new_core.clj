(ns jutsu.ai.new-core         
  (:import [org.datavec.api.split FileSplit]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder]
           [org.nd4j.linalg.activations Activation]
           [org.deeplearning4j.nn.conf.layers OutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.util ModelSerializer]
           [org.deeplearning4j.nn.conf.layers RBM$Builder]
           [org.deeplearning4j.nn.conf.layers GravesLSTM$Builder]
           [org.deeplearning4j.eval Evaluation RegressionEvaluation]
           [org.deeplearning4j.nn.conf.layers RnnOutputLayer$Builder]
           [org.datavec.api.records.reader.impl.csv CSVSequenceRecordReader]
           [org.deeplearning4j.datasets.datavec SequenceRecordReaderDataSetIterator]))

(defn network-config [input]
  (NeuralNetConfiguration$Builder.))

(defn translate-to-java [key]
  (let [tokens (clojure.string/split (name key) #"-")
        t0 (first tokens)]
    (str "." t0 (apply str (mapv clojure.string/capitalize (rest tokens))))))

(defn get-layers-key-index [ks]
  (let [index (.indexOf ks :layers)]
    (if (not= -1 index) 
      (if (> index 0) index
        (throw (Exception. ":layers key cannot be at zero index")))
      (throw (Exception. ":layers key not found in config")))))

(defn init-config-parse [edn-config]
  (let [ks (keys edn-config)
        layers-index (get-layers-key-index ks)]
    (split-at layers-index (map
                             (fn [k] [(translate-to-java k) 
                                      (get edn-config k)])
                             ks))))

(def options
  {:sgd (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :tanh (Activation/TANH)
   :identity (Activation/IDENTITY)})

(defn get-option [arg]
  (let [option (get options arg)]
    (if (nil? option)
      (throw (Exception. (str arg " is not an option")))
      option)))

(defn parse-arg [arg] 
  (if (keyword? arg) (get-option arg) arg))

(defmacro parse-option 
  ([option] (list 'fn '[net] (list option 'net)))
  ([option arg] (list 'fn '[net] (list option 'net arg))))

(defn branch-config [parsed-config]
  (let [header (first parsed-config)
        body-footer (split-at 1 (second parsed-config))
        body (first body-footer)
        footer (second body-footer)]
    (map (fn [el]
           (let [option (first el)
                 arg (second el)]
             (parse-option (symbol option) arg))) 
      header)))
    
(defn network [edn-config]
  (-> edn-config
      init-config-parse
      branch-config))

(def netty (NeuralNetConfiguration$Builder.))
