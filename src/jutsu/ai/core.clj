(ns jutsu.ai.core           
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

(defn regression-csv-iterator [filename batch-size label-index]
  (let [path (-> (ClassPathResource. filename)
              (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr nil batch-size label-index -1 true)))

(defn classification-csv-iterator [filename batch-size label-index num-possible-labels]
  (let [path (-> (ClassPathResource. filename)
                 (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr batch-size label-index num-possible-labels)))

(defn sequence-regression-csv-iterator 
  "Use for Recurrent Neural Nets"
  [filename batch-size label-index]
  (let [path (-> (ClassPathResource. filename)
              (.getFile))
        rr (CSVSequenceRecordReader. 0 ";")]
    (.initialize rr (FileSplit. path))
    (SequenceRecordReaderDataSetIterator. rr batch-size -1 label-index true)))

(defn sequence-classification-csv-iterator
  "Use for Recurrent Neural Nets"
  [filename batch-size label-index num-possible-labels]
  (let [path (-> (ClassPathResource. filename)
                 (.getFile))
        rr (CSVSequenceRecordReader. 0 ";")]
    (.initialize rr (FileSplit. path))
    (SequenceRecordReaderDataSetIterator. rr batch-size num-possible-labels label-index)))

(defn translate-to-java [key]
  (let [tokens (clojure.string/split (name key) #"-")
        t0 (first tokens)]
    (str t0 (apply str (mapv clojure.string/capitalize (rest tokens))))))

(defn get-layers-key-index [ks]
  (let [index (.indexOf ks :layers)]
    (if (not= -1 index) index
      ;(if (> index 0) index
      ;  (throw (Exception. ":layers key cannot be at zero index"))
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
   :identity (Activation/IDENTITY)
   :mse (LossFunctions$LossFunction/MSE)
   :negative-log-likelihood (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
   :kl-divergence (LossFunctions$LossFunction/KL_DIVERGENCE)
   :relu (Activation/RELU)
   :softmax (Activation/SOFTMAX)
   :sigmoid (Activation/SIGMOID)})

(defn get-option [arg]
  (let [option (get options arg)]
    (if (nil? option)
      (throw (Exception. (str arg " is not a supported justu.ai option")))
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
    (fn [net]
      (str-invoke net method arg))))

(def layer-builders
  {:dense (fn [] (DenseLayer$Builder.))
   :rbm (fn [] (RBM$Builder.))
   :graves-lstm (fn [] (GravesLSTM$Builder.))
   :output (fn [loss-fn] (OutputLayer$Builder. loss-fn))
   :rnn-output (fn [loss-fn] (RnnOutputLayer$Builder. loss-fn))})

(defn prepare-layer-config [layer-config]
  (->> (map (fn [k] [k (get layer-config k)]) (keys layer-config))
       (map parse-element)
       (apply comp)))

(defn normal-layer [i layer-builder config]
  (let [config-methods (prepare-layer-config config)]
    (fn [net]
      (.layer net i (-> (layer-builder)
                        config-methods
                        .build)))))

(defn special-loss-layer [i layer-builder config]
  (let [loss-fn (parse-arg (get config :loss))
        config-methods (prepare-layer-config (dissoc config :loss))]
    (fn [net]
      ;(println net)
      ;(println loss-fn)
      (.layer net i (-> (layer-builder loss-fn)
                        config-methods
                        .build)))))
    
;;produce a transducer to call on the layer builder
;;layers with loss can be special
;;look for loss keyword
;;need to create special case for output layer
(defn parse-layer [i [layer-type layer-config]]
  (let [layer-builder (get layer-builders layer-type)]
    (if (contains? layer-config :loss)
      (special-loss-layer i layer-builder layer-config)
      (normal-layer i layer-builder layer-config))))

(defn parse-body [body]
  (doall (map-indexed (fn [i layer] (parse-layer i layer)) body)))

;;Order of header-body-footer matters
;;builds a transducer of instance methods to call on the neural net object
(defn branch-config [parsed-config]
  (let [header (first parsed-config)
        body-footer (split-at 1 (second parsed-config))
        body (second (ffirst body-footer))
        footer (second body-footer)
        header-transducer (apply comp (map parse-element header))
        layers-transducer (apply comp (reverse (parse-body body)))
        footer-transducer (apply comp (map parse-element footer))]
    (fn [net]
      (-> net
          header-transducer
          .list
          layers-transducer
          footer-transducer
          .build))))
          
(defn config-network [edn-config]
  (let [network-transducer (-> edn-config
                               init-config-parse;;split config at layers index
                               branch-config)]
    (network-transducer (NeuralNetConfiguration$Builder.))))

(defn initialize-net!
  ([net-config]
   (initialize-net! net-config (list (ScoreIterationListener. 1))))
  ([net-config listeners]
   (let [net (MultiLayerNetwork. net-config)]
     (.init net)
     (.setListeners net listeners)
     net)))

;;set true for online learning
(defn save-model 
  ([net filename]
   (save-model net filename false))
  ([net filename ready-for-more]
   (ModelSerializer/writeModel net (java.io.File. filename) ready-for-more)))

(defn load-model [filename]
   (ModelSerializer/restoreMultiLayerNetwork filename))

(defn train-net! [net epochs dataset-iterator]
  (doseq [n (range 0 epochs)]
    (.reset dataset-iterator)
    (.fit net dataset-iterator))
  net)

(defn output
  ([net data] (.output net data false))
  ([net data train?] (.output net data train?)))

(defn evaluate [net dataset-iterator]
  (.stats (.evaluate net dataset-iterator)))

(defn evaluate-regression [net dataset-iterator]
  (.stats (.evaluateRegression net dataset-iterator)))
