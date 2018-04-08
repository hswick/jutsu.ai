(ns jutsu.ai.core
  (:import [org.datavec.api.split FileSplit]
           [org.datavec.api.util ClassPathResource]
           [org.datavec.api.io.labels ParentPathLabelGenerator]
           [org.datavec.image.recordreader ImageRecordReader]
           [org.nd4j.linalg.dataset.api.preprocessor ImagePreProcessingScaler]
           [org.datavec.image.loader NativeImageLoader]
           [org.deeplearning4j.datasets.datavec
            RecordReaderDataSetIterator
            SequenceRecordReaderDataSetIterator]
           [org.datavec.api.records.reader.impl.csv
            CSVRecordReader
            CSVSequenceRecordReader]
           [org.deeplearning4j.nn.conf
            NeuralNetConfiguration$Builder
            GradientNormalization
            ]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.util ModelSerializer]
           [org.deeplearning4j.nn.conf
            BackpropType
            WorkspaceMode]
           [org.deeplearning4j.nn.conf.layers
            SubsamplingLayer$Builder
            SubsamplingLayer$PoolingType
            ConvolutionLayer$Builder
            RnnOutputLayer$Builder
            GravesLSTM$Builder
            LSTM$Builder
            DropoutLayer$Builder
            OutputLayer$Builder
            DenseLayer$Builder
            LocalResponseNormalization$Builder GravesBidirectionalLSTM$Builder]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.conf.distribution
            NormalDistribution
            GaussianDistribution]
           [java.io File]
           [java.util Random]
           (org.nd4j.linalg.schedule StepSchedule MapSchedule ScheduleType ExponentialSchedule InverseSchedule PolySchedule SigmoidSchedule)
           (org.deeplearning4j.nn.conf.layers.variational VariationalAutoencoder VariationalAutoencoder$Builder)
           (org.nd4j.linalg.learning.config Nesterovs Adam RmsProp Sgd AdaDelta AdaGrad AdaMax Nadam NoOp)))

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

(defn classification-dir-labeled-image-iterator 
  [directory height width channels batch-size num-possible-labels rnd]
  (let [dir (File. directory)
        file-split (FileSplit. dir NativeImageLoader/ALLOWED_FORMATS (Random. rnd))         
        rr (ImageRecordReader. height width (ParentPathLabelGenerator.))
        _ (.initialize rr file-split) 
        data-iterator (RecordReaderDataSetIterator. rr batch-size 1 num-possible-labels) 
        data-normalizer (ImagePreProcessingScaler. 0 1)
        _ (.fit data-normalizer data-iterator)
        _ (.setPreProcessor data-iterator data-normalizer)]
    data-iterator))

(defn translate-to-java [key]
  (let [tokens (clojure.string/split (name key) #"-")
        t0 (first tokens)]
    (str t0 (apply str (mapv clojure.string/capitalize (rest tokens))))))

(defn get-layers-index [ks]
  (let [index (.indexOf ks :layers)]
    (if (not= -1 index)
      (if (= index 0) index
        (if (= index 1) 0
          (if (= index 2) 1
            (/ index 2))))
      (throw (Exception. ":layers key not found in config")))))

(defn init-config-parse [edn-config]
  (let [layers-index (get-layers-index edn-config)
        split-config (split-at layers-index (partition 2 edn-config))]
    split-config))

(def options
  {:sgd                      (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :tanh                     (Activation/TANH)
   :identity                 (Activation/IDENTITY)
   :mse                      (LossFunctions$LossFunction/MSE)
   :negative-log-likelihood  (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
   :kl-divergence            (LossFunctions$LossFunction/KL_DIVERGENCE)
   :relu                     (Activation/RELU)
   :softmax                  (Activation/SOFTMAX)
   :sigmoid                  (Activation/SIGMOID)
   :softsign                 (Activation/SOFTSIGN)
   :xavier                   (WeightInit/XAVIER)
   :mcxent                   (LossFunctions$LossFunction/MCXENT)
   :truncated-bptt           (BackpropType/TruncatedBPTT)
   :pooling-type-max         (SubsamplingLayer$PoolingType/MAX)
   :distribution             (WeightInit/DISTRIBUTION)
   :renormalize-l2-per-layer (GradientNormalization/RenormalizeL2PerLayer)
   :workspace-single         (WorkspaceMode/SINGLE)
   :workspace-separate       (WorkspaceMode/SEPARATE)
   })

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
    (if (= clojure.lang.PersistentVector (class arg))
      (fn [net]
       (str-invoke net method (int-array arg)))
      (fn [net]
        (str-invoke net method arg)))))

(def layer-builders
  {:dense                        (fn [] (DenseLayer$Builder.))
   :dropout                      (fn [dropout] (DropoutLayer$Builder. dropout))
   :graves-bidirectional-lstm    (fn [] (GravesBidirectionalLSTM$Builder.))
   :graves-lstm                  (fn [] (GravesLSTM$Builder.))
   :lstm                         (fn [] (LSTM$Builder.))
   :output                       (fn [loss-fn] (OutputLayer$Builder. loss-fn))
   :rnn-output                   (fn [loss-fn] (RnnOutputLayer$Builder. loss-fn))
   :vae                          (fn [] (VariationalAutoencoder$Builder.))
   :convolution                  (fn
                                   ([kernel-size] (ConvolutionLayer$Builder. (int-array kernel-size)))
                                   ([kernel stride]
                                    (ConvolutionLayer$Builder.
                                      (int-array kernel)
                                      (int-array stride)))
                                   ([kernel stride pad]
                                    (ConvolutionLayer$Builder.
                                      (int-array kernel)
                                      (int-array stride)
                                      (int-array pad))))
   :sub-sampling                 (fn
                                   ([pooling-type]
                                    (SubsamplingLayer$Builder. pooling-type))
                                   ([kernel-size stride]
                                    (SubsamplingLayer$Builder. (int-array kernel-size) (int-array stride))))
   :local-response-normalization (fn [] (LocalResponseNormalization$Builder.))})

(defn prepare-layer-config [layer-config]
  (->> (partition 2 layer-config)
       (map parse-element)
       (apply comp)))

(defn normal-layer [i layer-builder config]
  (let [config-methods (prepare-layer-config config)]
    (fn [net]
      (.layer net i (-> (layer-builder)
                        config-methods
                        .build)))))

(defn special-layer [i layer-builder config]
  (let [config-methods (prepare-layer-config (last config))]
    (fn [net]
      (.layer net i (-> (apply layer-builder (map parse-arg (drop-last config)))
                        config-methods
                        .build)))))

(defn parse-layer [i layer]
  (let [layer-type (first layer)
        layer-config (rest layer)
        layer-builder (get layer-builders layer-type)]
    (if (= 1 (count layer-config))
      (normal-layer i layer-builder (first layer-config))
      (special-layer i layer-builder layer-config))))

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
          
(defn network [edn-config]
  (let [network-transducer (-> edn-config
                               init-config-parse;;split config at layers index
                               branch-config)]
    (MultiLayerNetwork. (network-transducer (NeuralNetConfiguration$Builder.)))))

(defn initialize-net
  ([net-config]
   (initialize-net net-config (list (ScoreIterationListener. 1))))
  ([net-config listeners]
   (let [net net-config]
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

(defn input-type-convolutional-flat [arg1 arg2 arg3]
  (InputType/convolutionalFlat arg1 arg2 arg3))

(defn input-type-convolutional [arg1 arg2 arg3]
  (InputType/convolutional arg1 arg2 arg3))

(defn normal-distribution [min max]
  (NormalDistribution. min max))

(defn guassian-distribution [min max]
  (GaussianDistribution. min max))

(defmulti build-updater  (fn [updater-key opts] updater-key))

(defmethod build-updater :rms-prop
  [_ {:keys [learning-rate learning-rate-schedule epsilon]}]
  (cond
    (and learning-rate learning-rate-schedule epsilon) (RmsProp. learning-rate learning-rate-schedule epsilon)
    learning-rate (RmsProp. learning-rate)
    learning-rate-schedule (RmsProp. learning-rate-schedule)
    :else (RmsProp.)
    ))

(defmethod build-updater :adam
  [_ {:keys [learning-rate learning-rate-schedule beta1 beta2 epsilon]}]
  (cond
    (and learning-rate beta1 beta2 epsilon) (Adam. learning-rate beta1 beta2 epsilon)
    learning-rate (Adam. learning-rate)
    learning-rate-schedule (Adam. learning-rate-schedule)
    :else (Adam.)
    ))

(defmethod build-updater :nesterovs
  [_ {:keys [learning-rate momentum-schedule learning-rate-schedule momentum]}]
  (cond
    (and learning-rate momentum-schedule) (Nesterovs. learning-rate momentum-schedule)
    (and learning-rate-schedule momentum-schedule) (Nesterovs. learning-rate-schedule momentum-schedule)
    (and learning-rate-schedule momentum) (Nesterovs. learning-rate-schedule momentum)
    learning-rate-schedule (Nesterovs. learning-rate-schedule)
    (and learning-rate momentum) (Nesterovs. learning-rate momentum)
    momentum (Nesterovs. momentum)
    :else (Nesterovs.)
    ))

(defmethod build-updater :sgd
  [_ {:keys [learning-rate-schedule learning-rate]}]
  (cond
    learning-rate-schedule (Sgd. learning-rate-schedule)
    learning-rate (Sgd. learning-rate)
    :else (Sgd.)))

(defmethod build-updater :ada-delta
  [_ {:keys []}]
  (AdaDelta.))

(defmethod build-updater :ada-grad
  [_ {:keys [learning-rate learning-rate-schedule epsilon]}]
  (cond
    (and learning-rate-schedule epsilon) (AdaGrad. learning-rate-schedule epsilon)
    learning-rate-schedule (AdaGrad. learning-rate-schedule)
    (and learning-rate epsilon) (AdaGrad. learning-rate epsilon)
    learning-rate (AdaGrad. learning-rate)
    :else (AdaGrad.)
    ))

(defmethod build-updater :ada-max
  [_ {:keys [learning-rate learning-rate-schedule beta1 beta2 epsilon]}]
  (cond
    (and learning-rate beta1 beta2 epsilon) (AdaMax. learning-rate beta1 beta2 epsilon)
    learning-rate-schedule (AdaMax. learning-rate-schedule)
    learning-rate (AdaMax. learning-rate)
    :else (AdaMax.)
    ))

(defmethod build-updater :nadam
  [_ {:keys [learning-rate learning-rate-schedule beta1 beta2 epsilon]}]
  (cond
    (and learning-rate beta1 beta2 epsilon) (Nadam. learning-rate beta1 beta2 epsilon)
    learning-rate-schedule (Nadam. learning-rate-schedule)
    learning-rate (Nadam. learning-rate)
    :else (Nadam.)
    ))

(defmethod build-updater :no-op
  [_ {:keys []}]
  (NoOp.))

(def schedule-type-map
  {:iteration (ScheduleType/ITERATION)
   :epoch (ScheduleType/EPOCH)})

(defmulti build-schedule-internal (fn [schedule-key opts] schedule-key))

(defmethod build-schedule-internal :exponential
  [_ {:keys [schedule-type initial-value gamma]}]
  (ExponentialSchedule. schedule-type (double initial-value) (double gamma)))

(defmethod build-schedule-internal :inverse
  [_ {:keys [schedule-type initial-value gamma power]}]
  (InverseSchedule. schedule-type (double initial-value) (double gamma) (double power)))

(defmethod build-schedule-internal :map
  [_ {:keys [schedule-type values]}]
  (MapSchedule. schedule-type values))

(defmethod build-schedule-internal :poly
  [_ {:keys [schedule-type initial-value power max-iter]}]
  (PolySchedule. schedule-type (double initial-value) (double power) (int max-iter)))

(defmethod build-schedule-internal :sigmoid
  [_ {:keys [schedule-type initial-value gamma step-size]}]
  (SigmoidSchedule. schedule-type (double initial-value) (double gamma) (int step-size)))

(defmethod build-schedule-internal :step
  [_ {:keys [schedule-type initial-value decay-rate step]}]
  (StepSchedule. schedule-type (double initial-value) (double decay-rate) (double step)))

(defn build-schedule [schedule-type opts]
  (let [updated-opts (-> opts
                       (assoc :schedule-type (get schedule-type-map (:schedule-type opts))))]
    (build-schedule-internal schedule-type updated-opts)))