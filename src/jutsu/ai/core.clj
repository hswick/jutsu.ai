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
           [org.deeplearning4j.util ModelSerializer]))

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

(defn default-regression-options []
  {:seed 12345
   :iterations 1
   :optimization-algo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :learning-rate 0.01
   :weight-init (WeightInit/XAVIER)
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :activation (Activation/RELU)
   :pretrain false
   :backprop true
   :num-hidden-nodes 50
   :output-activation (Activation/IDENTITY)
   :loss-function (LossFunctions$LossFunction/MSE)})

(defn regression-net
  ([num-in num-out] (regression-net num-in num-out {}))
  ([num-in num-out options-map]
   (let [final-map (merge (default-regression-options) options-map)]
    (-> (NeuralNetConfiguration$Builder.)
      (.seed (:seed final-map))
      (.iterations (:iterations final-map))
      (.optimizationAlgo (:optimization-algo final-map))
      (.learningRate (:learning-rate final-map))
      (.weightInit (:weight-init final-map))
      (.updater (:updater final-map))
      (.momentum (:momentum final-map))
      (.list)
      (.layer 0 (-> (DenseLayer$Builder.)
                    (.nIn num-in)
                    (.nOut (:num-hidden-nodes final-map))
                    (.activation (:activation final-map))
                    (.build)))
      (.layer 1 (-> (DenseLayer$Builder.)
                    (.nIn (:num-hidden-nodes final-map))
                    (.nOut (:num-hidden-nodes final-map))
                    (.activation (:activation final-map))
                    (.build)))
      (.layer 2 (-> (OutputLayer$Builder. (:loss-function final-map))
                    (.activation (:output-activation final-map))
                    (.nIn (:num-hidden-nodes final-map))
                    (.nOut num-out)
                    (.build)))
      (.pretrain (:pretrain final-map))
      (.backprop (:backprop final-map))
      (.build)
      (MultiLayerNetwork.)))))

(defn default-classification-options []
  {:seed 12345
   :optimization-algo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :iterations 1
   :learning-rate 0.006
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :regularization true
   :l2 1e-4
   :weight-init (WeightInit/XAVIER)
   :activation (Activation/RELU)
   :loss-function (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
   :output-activation (Activation/SOFTMAX)
   :pretrain false
   :backprop true})

(defn classification-net
  ([num-in num-out] (classification-net num-in num-out {}))
  ([num-in num-out options-map]
   (let [final-map (merge (default-classification-options) options-map)]
     (-> (NeuralNetConfiguration$Builder.)
       (.seed (:seed final-map))
       (.optimizationAlgo (:optimization-algo final-map))
       (.iterations (:iterations final-map))
       (.learningRate (:learning-rate final-map))
       (.updater (:updater final-map))
       (.momentum (:momentum final-map))
       (.regularization (:regularization final-map))
       (.l2 (:l2 final-map))
       (.list)
       (.layer 0 (-> (DenseLayer$Builder.)
                     (.nIn num-in)
                     (.nOut (:num-hidden-nodes final-map))
                     (.activation (:activation final-map))
                     (.weightInit (:weight-init final-map))
                     (.build)))
       (.layer 1 (-> (OutputLayer$Builder. (:loss-function final-map))
                     (.nIn (:num-hidden-nodes final-map))
                     (.nOut num-out)
                     (.activation (:output-activation final-map))
                     (.weightInit (:weight-init final-map))
                     (.build)))
       (.pretrain (:pretrain final-map))
       (.backprop (:backprop final-map))
       (.build)
       (MultiLayerNetwork.)))))

;;set true for online learning
(defn save-model 
  ([net filename]
   (save-model net filename false))
  ([net filename ready-for-more]
   (ModelSerializer/writeModel net (java.io.File. filename) ready-for-more)))

(defn load-model [filename]
   (ModelSerializer/restoreMultiLayerNetwork filename))

(defn initialize-net!
  ([net] (initialize-net! net (list (ScoreIterationListener. 1))))
  ([net listeners]
   (.init net)
   (.setListeners net listeners)
   net))

(defn train-net! [net epochs dataset-iterator]
  (doseq [n (range 0 epochs)]
    (.reset dataset-iterator)
    (.fit net dataset-iterator))
  net)

(defn parse-shorthand [topology]
  (map (fn [layer]
         {:in (nth layer 0)
          :out (nth layer 1)
          :activation (nth layer 2)})
    topology))

(def activation-options
  {:relu (Activation/RELU)
   :identity (Activation/IDENTITY)
   :soft-max (Activation/SOFTMAX)
   :tanh (Activation/TANH)})

;;throw error if now activation key
(defn interpret-layer [i layer topo-count options-map]
    (fn [net]
      (-> net
        (.layer i (-> (if (= i topo-count)
                          (OutputLayer$Builder. (:loss-function options-map))
                          (DenseLayer$Builder.))
                      (.nIn (:in layer))
                      (.nOut (:out layer))
                      (.activation (get activation-options (:activation (:layer))))
                      (.build))))))

;;pass in shorthand with vectors
;;Returns a vector of functions to be called on the net
(defn parse-topology [topology options-map]
  (let [topo (if (= clojure.lang.PersistentVector (class (first topology)))
               (parse-shorthand topology)
               topology)
        topo-count (count topo)
        proper-topo (map-indexed (fn [i layer]
            (interpret-layer i layer (dec topo-count)) topo))]))

(defn initialize-layers [net parsed-topology]
  (doseq [layer-fn parsed-topology]
    (layer-fn net))
  net)

(defn network [topology options-map]
  (let [parsed-topology (parse-topology topology options-map)]
    (-> (NeuralNetConfiguration$Builder.)
      (.seed (:seed options-map))
      (.iterations (:iterations options-map))
      (.optimizationAlgo (:optimization-algo options-map))
      (.learningRate (:learning-rate options-map))
      (.weightInit (:weight-init options-map))
      (.updater (:updater options-map))
      (.momentum (:momentum options-map))
      (.list)
      (initialize-layers parsed-topology)
      (.pretrain (:pretrain options-map))
      (.backprop (:backprop options-map))
      (.build)
      (MultiLayerNetwork.))))

