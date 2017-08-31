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

(defn translate-to-java [key]
  (let [tokens (clojure.string/split (name key) #"-")
        t0 (first tokens)]
    (str t0 (apply str (mapv clojure.string/capitalize (rest tokens))))))

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
                             (fn [k] [k 
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

;;from https://en.wikibooks.org/wiki/Clojure_Programming/Examples#Invoking_Java_method_through_method_name_as_a_String
(defn str-invoke [instance method-str & args]
            (clojure.lang.Reflector/invokeInstanceMethod 
                instance 
                method-str 
                (to-array args)))

(defn parse-element [el]
  (let [method (translate-to-java (first el))
        arg (parse-arg (second el))]
    (fn [net] (str-invoke net method arg))))

(def layer-builders
  {:default (fn [] (DenseLayer$Builder.))
   :rbm (fn [] (RBM$Builder.))
   :graves-lstm (fn [] (GravesLSTM$Builder.))
   :output (fn [loss-fn] (OutputLayer$Builder. loss-fn))
   :rnn-output (fn [loss-fn] (RnnOutputLayer$Builder. loss-fn))})

(defn prepare-layer-config [layer-config]
  (->> (map (fn [k] [k (get layer-config k)]) (keys layer-config))
       (map parse-element)
       (apply comp)))

;;produce a transducer to call on the layer builder
;;layers with loss can be special
;;look for loss keyword
(defn parse-layer [i [layer-type layer-config]]
  (let [layer-builder (get layer-builders layer-type)
        config-methods (prepare-layer-config layer-config)]
    (fn [net] (.layer net i 
                ((comp config-methods (fn [layer] (.build layer))) layer-builder)))))

(defn parse-body [body]
  (map-indexed (fn [i layer] (parse-layer i layer)) body))

;;Order of header-body-footer matters
;;builds a transducer of instance methods to call on the neural net object
(defn branch-config [parsed-config]
  (let [header (first parsed-config)
        body-footer (split-at 1 (second parsed-config))
        body (second (ffirst body-footer))
        footer (second body-footer)]
    [(apply comp (map parse-element header))
     (fn [net] (.list net))
     (apply comp (parse-body body))
     (apply comp (map parse-element footer))
     (fn [net] (.build net))]))

(defn network [edn-config]
  (-> edn-config
      init-config-parse;;split config at layers index
      branch-config
      ((fn [config] (apply comp config)))))

(def netty (NeuralNetConfiguration$Builder.))
