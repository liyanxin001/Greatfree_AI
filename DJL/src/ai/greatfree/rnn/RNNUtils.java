package ai.greatfree.rnn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.logging.Logger;

import org.greatfree.util.Time;

import com.ibm.icu.util.Calendar;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.greatfree.utils.Accumulator;
import ai.greatfree.utils.Functions;
import ai.greatfree.utils.Functions.SimpleFunction;
import ai.greatfree.utils.Functions.voidTwoFunction;
import ai.greatfree.utils.RNNModelScratch;
import ai.greatfree.utils.StopWatch;
import ai.greatfree.utils.Training;
import ai.greatfree.utils.Vocab;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

/**
 * Created: 08/27/2024, Bing Li
 */
final class RNNUtils
{
	public static Figure plot(double[] x, double[] y, String xLabel, String yLabel)
	{
		ScatterTrace trace = ScatterTrace.builder(x, y).mode(ScatterTrace.Mode.LINE).build();
		Layout layout = Layout.builder().showLegend(true).xAxis(Axis.builder().title(xLabel).build()).yAxis(Axis.builder().title(yLabel).build()).build();
		return new Figure(layout, trace);
	}

	// A simple MLP
	public static SequentialBlock getNet()
	{
		SequentialBlock net = new SequentialBlock();
		net.add(Linear.builder().setUnits(10).build());
		net.add(Activation::relu);
		net.add(Linear.builder().setUnits(1).build());
		return net;
	}

	public static Model train(SequentialBlock net, ArrayDataset dataset, int batchSize, int numEpochs, float learningRate) throws IOException, TranslateException
	{
		// Square Loss
		Loss loss = Loss.l2Loss();
		Tracker lrt = Tracker.fixed(learningRate);
		Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

		DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(adam) // Optimizer (loss function)
				.optInitializer(new XavierInitializer(), "").addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

		Model model = Model.newInstance("sequence");
		model.setBlock(net);
		Trainer trainer = model.newTrainer(config);

		for (int epoch = 1; epoch <= numEpochs; epoch++)
		{
			// Iterate over dataset
			for (Batch batch : trainer.iterateDataset(dataset))
			{
				// Update loss and evaulator
				EasyTrain.trainBatch(trainer, batch);
				// Update parameters
				trainer.step();
				batch.close();
			}
			// reset training and validation evaluators at end of epoch
			trainer.notifyListeners(listener -> listener.onEpoch(trainer));
			System.out.printf("Epoch %d\n", epoch);
			System.out.printf("Loss %f\n", trainer.getTrainingResult().getTrainLoss());
		}
		return model;
	}

	public static Figure oneStepPredict(NDArray oneStepPreds, NDArray time, NDArray x, int tau) throws TranslateException
	{
		ScatterTrace trace = ScatterTrace.builder(Functions.floatToDoubleArray(time.toFloatArray()), Functions.floatToDoubleArray(x.toFloatArray())).mode(ScatterTrace.Mode.LINE).name("data").build();
		ScatterTrace trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:", tau)).toFloatArray()), Functions.floatToDoubleArray(oneStepPreds.toFloatArray())).mode(ScatterTrace.Mode.LINE).name("1-step preds").build();

		Layout layout = Layout.builder().showLegend(true).xAxis(Axis.builder().title("time").build()).yAxis(Axis.builder().title("x").build()).build();
		return new Figure(layout, trace, trace2);
	}

	public static Figure multiStepPredict(NDArray multiStepPreds, NDArray oneStepPreds, int nTrain, NDArray time, NDArray x, int tau) throws TranslateException
	{
		ScatterTrace trace = ScatterTrace.builder(Functions.floatToDoubleArray(time.toFloatArray()), Functions.floatToDoubleArray(x.toFloatArray())).mode(ScatterTrace.Mode.LINE).name("data").build();
		ScatterTrace trace2 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:", tau)).toFloatArray()), Functions.floatToDoubleArray(oneStepPreds.toFloatArray())).mode(ScatterTrace.Mode.LINE).name("1-step preds").build();
		ScatterTrace trace3 = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:", nTrain + tau)).toFloatArray()), Functions.floatToDoubleArray(multiStepPreds.get(new NDIndex("{}:", nTrain + tau)).toFloatArray())).mode(ScatterTrace.Mode.LINE).name("multistep preds").build();

		Layout layout = Layout.builder().showLegend(true).xAxis(Axis.builder().title("time").build()).yAxis(Axis.builder().title("x").build()).build();
		return new Figure(layout, trace, trace2, trace3);
	}

	public static Figure multiStepPredict(NDArray time, NDArray features, int maxSteps, int tau, int T)
	{
		int[] steps = new int[] { 1, 4, 16, 64 };

		ScatterTrace[] traces = new ScatterTrace[4];

		for (int i = 0; i < traces.length; i++)
		{
			int step = steps[i];
			traces[i] = ScatterTrace.builder(Functions.floatToDoubleArray(time.get(new NDIndex("{}:{}", tau + step - 1, T - maxSteps + i)).toFloatArray()), Functions.floatToDoubleArray(features.get(new NDIndex(":,{}", tau + step - 1)).toFloatArray())).mode(ScatterTrace.Mode.LINE).name(step + "-step preds").build();
		}

		Layout layout = Layout.builder().showLegend(true).xAxis(Axis.builder().title("time").build()).yAxis(Axis.builder().title("x").build()).build();
		return new Figure(layout, traces);
	}

	public static String[] readTimeMachine() throws IOException
	{
		@SuppressWarnings("deprecation")
		URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
		String[] lines;
		try (BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream())))
		{
			lines = in.lines().toArray(String[]::new);
		}

		for (int i = 0; i < lines.length; i++)
		{
			lines[i] = lines[i].replaceAll("[^A-Za-z]+", " ").strip().toLowerCase();
		}
		return lines;
	}

	public static String[][] tokenize(String[] lines, String token) throws Exception
	{
		// Split text lines into word or character tokens.
		String[][] output = new String[lines.length][];
		if (token == "word")
		{
			for (int i = 0; i < output.length; i++)
			{
				output[i] = lines[i].split(" ");
			}
		}
		else if (token == "char")
		{
			for (int i = 0; i < output.length; i++)
			{
				output[i] = lines[i].split("");
			}
		}
		else
		{
			throw new Exception("ERROR: unknown token type: " + token);
		}
		return output;
	}

	public static Pair<List<Integer>, Vocab> loadCorpusTimeMachine(int maxTokens) throws IOException, Exception
	{
		/* Return token indices and the vocabulary of the time machine dataset. */
		String[] lines = readTimeMachine();
		String[][] tokens = tokenize(lines, "char");
		Vocab vocab = new Vocab(tokens, 0, new String[0]);
		// Since each text line in the time machine dataset is not necessarily a
		// sentence or a paragraph, flatten all the text lines into a single list
		List<Integer> corpus = new ArrayList<>();
		for (int i = 0; i < tokens.length; i++)
		{
			for (int j = 0; j < tokens[i].length; j++)
			{
				if (tokens[i][j] != "")
				{
					corpus.add(vocab.getIdx(tokens[i][j]));
				}
			}
		}
		if (maxTokens > 0)
		{
			corpus = corpus.subList(0, maxTokens);
		}
		return new Pair<List<Integer>, Vocab>(corpus, vocab);
	}

	/**
	 * Generate a minibatch of subsequences using random sampling.
	 */
	public static ArrayList<NDList> seqDataIterRandom(List<Integer> corpus, int batchSize, int numSteps, NDManager manager)
	{
		// Start with a random offset (inclusive of `numSteps - 1`) to partition a
		// sequence
		corpus = corpus.subList(new Random().nextInt(numSteps - 1), corpus.size());
		// Subtract 1 since we need to account for labels
		int numSubseqs = (corpus.size() - 1) / numSteps;
		// The starting indices for subsequences of length `numSteps`
		List<Integer> initialIndices = new ArrayList<>();
		for (int i = 0; i < numSubseqs * numSteps; i += numSteps)
		{
			initialIndices.add(i);
		}
		// In random sampling, the subsequences from two adjacent random
		// minibatches during iteration are not necessarily adjacent on the
		// original sequence
		Collections.shuffle(initialIndices);

		int numBatches = numSubseqs / batchSize;

		ArrayList<NDList> pairs = new ArrayList<NDList>();
		for (int i = 0; i < batchSize * numBatches; i += batchSize)
		{
			// Here, `initialIndices` contains randomized starting indices for
			// subsequences
			// List<Integer> initialIndicesPerBatch = initialIndices.subList(i, i +
			// batchSize);
			NDArray xNDArray = manager.create(new Shape(initialIndices.size(), numSteps), DataType.INT32);
			NDArray yNDArray = manager.create(new Shape(initialIndices.size(), numSteps), DataType.INT32);
			for (int j = 0; j < initialIndices.size(); j++)
			{
				ArrayList<Integer> X = data(initialIndices.get(j), corpus, numSteps);
				xNDArray.set(new NDIndex(j), manager.create(X.stream().mapToInt(Integer::intValue).toArray()));
				ArrayList<Integer> Y = data(initialIndices.get(j) + 1, corpus, numSteps);
				yNDArray.set(new NDIndex(j), manager.create(Y.stream().mapToInt(Integer::intValue).toArray()));
			}
			NDList pair = new NDList();
			pair.add(xNDArray);
			pair.add(yNDArray);
			pairs.add(pair);
		}
		return pairs;
	}

	private static ArrayList<Integer> data(int pos, List<Integer> corpus, int numSteps)
	{
		// Return a sequence of length `numSteps` starting from `pos`
		return new ArrayList<Integer>(corpus.subList(pos, pos + numSteps));
	}

	/**
	 * Generate a minibatch of subsequences using sequential partitioning.
	 */
	public static ArrayList<NDList> seqDataIterSequential(List<Integer> corpus, int batchSize, int numSteps, NDManager manager)
	{
		// Start with a random offset to partition a sequence
		int offset = new Random().nextInt(numSteps);
		int numTokens = ((corpus.size() - offset - 1) / batchSize) * batchSize;

		NDArray Xs = manager.create(corpus.subList(offset, offset + numTokens).stream().mapToInt(Integer::intValue).toArray());
		NDArray Ys = manager.create(corpus.subList(offset + 1, offset + 1 + numTokens).stream().mapToInt(Integer::intValue).toArray());
		Xs = Xs.reshape(new Shape(batchSize, -1));
		Ys = Ys.reshape(new Shape(batchSize, -1));
		int numBatches = (int) Xs.getShape().get(1) / numSteps;

		ArrayList<NDList> pairs = new ArrayList<NDList>();
		for (int i = 0; i < numSteps * numBatches; i += numSteps)
		{
			NDArray X = Xs.get(new NDIndex(":, {}:{}", i, i + numSteps));
			NDArray Y = Ys.get(new NDIndex(":, {}:{}", i, i + numSteps));
			NDList pair = new NDList();
			pair.add(X);
			pair.add(Y);
			pairs.add(pair);
		}
		return pairs;
	}

	/**
	 * Return the iterator and the vocabulary of the time machine dataset.
	 */
	public static Pair<ArrayList<NDList>, Vocab> loadDataTimeMachine(int batchSize, int numSteps, boolean useRandomIter, int maxTokens, NDManager manager) throws IOException, Exception
	{
		SeqDataLoader seqData = new SeqDataLoader(batchSize, numSteps, useRandomIter, maxTokens, manager);
		return new Pair<ArrayList<NDList>, Vocab>(seqData.dataIter, seqData.vocab); // ArrayList<NDList>, Vocab
	}

	public static NDList getParams(int vocabSize, int numHiddens, Device device, NDManager manager)
	{
		int numOutputs = vocabSize;
		int numInputs = vocabSize;

		// Hidden layer parameters
		NDArray W_xh = normal(new Shape(numInputs, numHiddens), device, manager);
		NDArray W_hh = normal(new Shape(numHiddens, numHiddens), device, manager);
		NDArray b_h = manager.zeros(new Shape(numHiddens), DataType.FLOAT32, device);
		// Output layer parameters
		NDArray W_hq = normal(new Shape(numHiddens, numOutputs), device, manager);
		NDArray b_q = manager.zeros(new Shape(numOutputs), DataType.FLOAT32, device);

		// Attach gradients
		NDList params = new NDList(W_xh, W_hh, b_h, W_hq, b_q);
		for (NDArray param : params)
		{
			param.setRequiresGradient(true);
		}
		return params;
	}

	public static NDArray normal(Shape shape, Device device, NDManager manager)
	{
		return manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device);
	}

	public static NDList initRNNState(int batchSize, int numHiddens, Device device, NDManager manager)
	{
		return new NDList(manager.zeros(new Shape(batchSize, numHiddens), DataType.FLOAT32, device));
	}

	public static Pair<NDArray, NDList> rnn(NDArray inputs, NDList state, NDList params)
	{
		// Shape of `inputs`: (`numSteps`, `batchSize`, `vocabSize`)
		NDArray W_xh = params.get(0);
		NDArray W_hh = params.get(1);
		NDArray b_h = params.get(2);
		NDArray W_hq = params.get(3);
		NDArray b_q = params.get(4);
		NDArray H = state.get(0);

		NDList outputs = new NDList();
		// Shape of `X`: (`batchSize`, `vocabSize`)
		NDArray X, Y;
		for (int i = 0; i < inputs.size(0); i++)
		{
			X = inputs.get(i);
			H = (X.dot(W_xh).add(H.dot(W_hh)).add(b_h)).tanh();
			Y = H.dot(W_hq).add(b_q);
			outputs.add(Y);
		}
		return new Pair<>(outputs.size() > 1 ? NDArrays.concat(outputs) : outputs.get(0), new NDList(H));
	}

	/** Generate new characters following the `prefix`. */
	public static String predict(String prefix, int numPreds, RNNModelScratch net, Vocab vocab, Device device, NDManager manager)
	{
		NDList state = net.beginState(1, device);
		List<Integer> outputs = new ArrayList<>();
		outputs.add(vocab.getIdx("" + prefix.charAt(0)));
		SimpleFunction<NDArray> getInput = () -> manager.create(outputs.get(outputs.size() - 1)).toDevice(device, false).reshape(new Shape(1, 1));
		for (char c : prefix.substring(1).toCharArray())
		{ // Warm-up period
			state = (NDList) net.forward(getInput.apply(), state).getValue();
			outputs.add(vocab.getIdx("" + c));
		}

		NDArray y;
		for (int i = 0; i < numPreds; i++)
		{
			Pair<NDArray, NDList> pair = net.forward(getInput.apply(), state);
			y = pair.getKey();
			state = pair.getValue();
			outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
		}
		StringBuilder output = new StringBuilder();
		for (int i : outputs)
		{
			output.append(vocab.idxToToken.get(i));
		}
		return output.toString();
	}

	private final static Logger log = Logger.getLogger("ai.greatfree.rnn");

	/*
	 * To ease the design of gradients of aggregation, the method is created by me
	 * such that NDManager is split from the aggregation. 08/29/2024, Bing Li
	 */
	private static void associateWithMan(RNNModelScratch net, NDManager manager)
	{
		for (NDArray p : net.params)
		{
			NDArray gradient = p.getGradient();
			gradient.attach(manager);
		}
	}

	/**
	 * The method is updated to make the gradient aggregation conveniently.
	 * 08/29/2024, Bing Li
	 * 
	 * The method must be the way to aggregate gradients. 08/28/2024, Bing Li
	 */
	/** Clip the gradient. */
	// public static void gradClipping(RNNModelScratch net, int theta, NDManager
	// manager)
	public static void gradClipping(RNNModelScratch net, int theta)
	{
		double result = 0;
		for (NDArray p : net.params)
		{
			// log.info("param bytes size = " + p.encode().length);
			NDArray gradient = p.getGradient();
			// log.info("gradient bytes size = " + gradient.encode().length);

			/**
			 * The method, associateWithMan, is responsible for the below task. 08/29/2024,
			 * Bing Li
			 * 
			 * The below line is NOT necessary? 08/29/2024, Bing Li
			 */
			// gradient.attach(manager);
			result += gradient.pow(2).sum().getFloat();
		}
		double norm = Math.sqrt(result);
		if (norm > theta)
		{
			for (NDArray param : net.params)
			{
				NDArray gradient = param.getGradient();
				gradient.muli(theta / norm);
			}
		}
	}

	/** Train a model within one epoch. */
	public static Pair<Double, Double> trainEpoch(RNNModelScratch net, List<NDList> trainIter, Loss loss, voidTwoFunction<Integer, NDManager> updater, Device device, boolean useRandomIter, NDManager manager)
	{
		StopWatch watch = new StopWatch();
		watch.start();
		Accumulator metric = new Accumulator(2); // Sum of training loss, no. of tokens
		try (NDManager childManager = manager.newSubManager())
		{
			NDList state = null;
			for (NDList pair : trainIter)
			{
				NDArray X = pair.get(0).toDevice(device, true);
				X.attach(childManager);
				NDArray Y = pair.get(1).toDevice(device, true);
				Y.attach(childManager);
				if (state == null || useRandomIter)
				{
					// Initialize `state` when either it is the first iteration or
					// using random sampling
					state = net.beginState((int) X.getShape().getShape()[0], device);
				}
				else
				{
					for (NDArray s : state)
					{
						s.stopGradient();
					}
				}
				state.attach(childManager);

				NDArray y = Y.transpose().reshape(new Shape(-1));
				X = X.toDevice(device, false);
				y = y.toDevice(device, false);
				try (GradientCollector gc = manager.getEngine().newGradientCollector())
				{
					Pair<NDArray, NDList> pairResult = net.forward(X, state);
					NDArray yHat = pairResult.getKey();
					state = pairResult.getValue();
					NDArray l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
					gc.backward(l);
					metric.add(new float[] { l.getFloat() * y.size(), y.size() });
				}

				associateWithMan(net, childManager);

				// gradClipping(net, 1, childManager);
				// Date startTime = Calendar.getInstance().getTime();
				gradClipping(net, 1);
				// Date endTime = Calendar.getInstance().getTime();
				// log.info("It takes " + Time.getTimespanInMilliSecond(endTime, startTime) + "
				// ms to finish gradients clipping!");

				updater.apply(1, childManager); // Since the `mean` function has been invoked
			}
		}
		return new Pair<>(Math.exp(metric.get(0) / metric.get(1)), metric.get(1) / watch.stop());
	}

	/** Train a model. */
	public static void train(RNNModelScratch net, List<NDList> trainIter, Vocab vocab, int lr, int numEpochs, Device device, boolean useRandomIter, NDManager manager)
	{
		SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
		// Animator animator = new Animator();
		// Initialize
		voidTwoFunction<Integer, NDManager> updater = (batchSize, subManager) -> Training.sgd(net.params, lr, batchSize, subManager);
		Function<String, String> predict = (prefix) -> predict(prefix, 50, net, vocab, device, manager);
		// Function<String, String> predict = (prefix) -> predict(prefix, 100, net,
		// vocab, device, manager);
		// Train and predict
		double ppl = 0.0;
		double speed = 0.0;
		for (int epoch = 0; epoch < numEpochs; epoch++)
		{
			/*
			 * for (NDList entry : trainIter) { // System.out.println(entry); for (NDArray
			 * dn : entry.getResourceNDArrays()) { System.out.println("dn length = " +
			 * dn.encode().length); System.out.println("dn = " + dn); } }
			 */

			Pair<Double, Double> pair = trainEpoch(net, trainIter, loss, updater, device, useRandomIter, manager);
			ppl = pair.getKey();
			speed = pair.getValue();
			if ((epoch + 1) % 10 == 0)
			{
				// animator.add(epoch + 1, (float) ppl, "");
				// animator.show();
				System.out.println("Epoch = " + (epoch + 1) + ", PPL = " + ppl);
			}
			// System.out.println("Press Enter to continue ...");
			// Tools.INPUT.nextLine();
		}
		System.out.format("perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
		Date startTime = Calendar.getInstance().getTime();
		System.out.println(predict.apply("time traveller"));
		Date endTime = Calendar.getInstance().getTime();
		log.info("1) It takes " + Time.getTimespanInMilliSecond(endTime, startTime) + " ms to predict ...");

		startTime = Calendar.getInstance().getTime();
		System.out.println(predict.apply("traveller"));
		endTime = Calendar.getInstance().getTime();
		log.info("2) It takes " + Time.getTimespanInMilliSecond(endTime, startTime) + " ms to predict ...");
	}

	/** Generate new characters following the `prefix`. */
	@SuppressWarnings("resource")
	public static String concisePredict(String prefix, int numPreds, Object net, Vocab vocab, Device device, NDManager manager)
	{
		List<Integer> outputs = new ArrayList<>();
		outputs.add(vocab.getIdx("" + prefix.charAt(0)));
		Functions.SimpleFunction<NDArray> getInput = () -> manager.create(outputs.get(outputs.size() - 1)).toDevice(device, false).reshape(new Shape(1, 1));

		if (net instanceof RNNModelScratch)
		{
			RNNModelScratch castedNet = (RNNModelScratch) net;
			NDList state = castedNet.beginState(1, device);

			for (char c : prefix.substring(1).toCharArray())
			{ // Warm-up period
				state = (NDList) castedNet.forward(getInput.apply(), state).getValue();
				outputs.add(vocab.getIdx("" + c));
			}

			NDArray y;
			for (int i = 0; i < numPreds; i++)
			{
				Pair<NDArray, NDList> pair = castedNet.forward(getInput.apply(), state);
				y = pair.getKey();
				state = pair.getValue();

				outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
			}
		}
		else
		{
			AbstractBlock castedNet = (AbstractBlock) net;
			NDList state = null;
			for (char c : prefix.substring(1).toCharArray())
			{ // Warm-up period
				if (state == null)
				{
					// Begin state
					state = castedNet.forward(new ParameterStore(manager, false), new NDList(getInput.apply()), false).subNDList(1);
				}
				else
				{
					state = castedNet.forward(new ParameterStore(manager, false), new NDList(getInput.apply()).addAll(state), false).subNDList(1);
				}
				outputs.add(vocab.getIdx("" + c));
			}

			NDArray y;
			for (int i = 0; i < numPreds; i++)
			{
				try (NDList pair = castedNet.forward(new ParameterStore(manager, false), new NDList(getInput.apply()).addAll(state), false))
				{
					y = pair.get(0);
					state = pair.subNDList(1);
				}
				outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
			}
		}

		StringBuilder output = new StringBuilder();
		for (int i : outputs)
		{
			output.append(vocab.idxToToken.get(i));
		}
		return output.toString();
	}

	/** Train a model. */
	public static void trainCh8(Object net, RandomAccessDataset dataset, Vocab vocab, int lr, int numEpochs, Device device, boolean useRandomIter, NDManager manager) throws IOException, TranslateException
	{
		SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
		// Animator animator = new Animator();

		Functions.voidTwoFunction<Integer, NDManager> updater;
		if (net instanceof RNNModelScratch)
		{
			RNNModelScratch castedNet = (RNNModelScratch) net;
			updater = (batchSize, subManager) -> Training.sgd(castedNet.params, lr, batchSize, subManager);
		}
		else
		{
			// Already initialized net
			AbstractBlock castedNet = (AbstractBlock) net;
			Model model = Model.newInstance("model");
			model.setBlock(castedNet);

			Tracker lrt = Tracker.fixed(lr);
			Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

			DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss
																								// function)
					.optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT) // setting the initializer
					.optDevices(Engine.getInstance().getDevices(1)) // setting the number of GPUs needed
					.addEvaluator(new Accuracy()) // Model Accuracy
					.addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

			Trainer trainer = model.newTrainer(config);
			updater = (batchSize, subManager) -> trainer.step();
		}

		Function<String, String> predict = (prefix) -> concisePredict(prefix, 50, net, vocab, device, manager);
		// Train and predict
		double ppl = 0.0;
		double speed = 0.0;
		for (int epoch = 0; epoch < numEpochs; epoch++)
		{
			Pair<Double, Double> pair = conciseTrainEpoch(net, dataset, loss, updater, device, useRandomIter, manager);
			ppl = pair.getKey();
			speed = pair.getValue();
			if ((epoch + 1) % 10 == 0)
			{
				// animator.add(epoch + 1, (float) ppl, "ppl");
				// animator.show();
				System.out.println("Epoch = " + (epoch + 1) + " PPL = " + (float) ppl);
			}
		}
		System.out.format("perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
		System.out.println(predict.apply("time traveller"));
		System.out.println(predict.apply("traveller"));
	}

	/** Train a model within one epoch. */
	@SuppressWarnings("resource")
	public static Pair<Double, Double> conciseTrainEpoch(Object net, RandomAccessDataset dataset, Loss loss, Functions.voidTwoFunction<Integer, NDManager> updater, Device device, boolean useRandomIter, NDManager manager) throws IOException, TranslateException
	{
		StopWatch watch = new StopWatch();
		watch.start();
		Accumulator metric = new Accumulator(2); // Sum of training loss, no. of tokens

		try (NDManager childManager = manager.newSubManager())
		{
			NDList state = null;
			for (Batch batch : dataset.getData(childManager))
			{
				NDArray X = batch.getData().head().toDevice(device, true);
				NDArray Y = batch.getLabels().head().toDevice(device, true);
				if (state == null || useRandomIter)
				{
					// Initialize `state` when either it is the first iteration or
					// using random sampling
					if (net instanceof RNNModelScratch)
					{
						state = ((RNNModelScratch) net).beginState((int) X.getShape().getShape()[0], device);
					}
				}
				else
				{
					for (NDArray s : state)
					{
						s.stopGradient();
					}
				}
				if (state != null)
				{
					state.attach(childManager);
				}

				NDArray y = Y.transpose().reshape(new Shape(-1));
				X = X.toDevice(device, false);
				y = y.toDevice(device, false);
				try (GradientCollector gc = Engine.getInstance().newGradientCollector())
				{
					NDArray yHat;
					if (net instanceof RNNModelScratch)
					{
						Pair<NDArray, NDList> pairResult = ((RNNModelScratch) net).forward(X, state);
						yHat = pairResult.getKey();
						state = pairResult.getValue();
					}
					else
					{
						NDList pairResult;
						if (state == null)
						{
							// Begin state
							pairResult = ((AbstractBlock) net).forward(new ParameterStore(manager, false), new NDList(X), true);
						}
						else
						{
							pairResult = ((AbstractBlock) net).forward(new ParameterStore(manager, false), new NDList(X).addAll(state), true);
						}
						yHat = pairResult.get(0);
						state = pairResult.subNDList(1);
					}

					NDArray l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
					gc.backward(l);
					metric.add(new float[] { l.getFloat() * y.size(), y.size() });
				}
				conciseGradClipping(net, 1, childManager);
				updater.apply(1, childManager); // Since the `mean` function has been invoked
			}
		}
		return new Pair<>(Math.exp(metric.get(0) / metric.get(1)), metric.get(1) / watch.stop());
	}

	/** Clip the gradient. */
	public static void conciseGradClipping(Object net, int theta, NDManager manager)
	{
		double result = 0;
		NDList params;
		if (net instanceof RNNModelScratch)
		{
			params = ((RNNModelScratch) net).params;
		}
		else
		{
			params = new NDList();
			for (Pair<String, Parameter> pair : ((AbstractBlock) net).getParameters())
			{
				params.add(pair.getValue().getArray());
			}
		}
		for (NDArray p : params)
		{
			NDArray gradient = p.getGradient().stopGradient();
			gradient.attach(manager);
			result += gradient.pow(2).sum().getFloat();
		}
		double norm = Math.sqrt(result);
		if (norm > theta)
		{
			for (NDArray param : params)
			{
				NDArray gradient = param.getGradient();
				gradient.muli(theta / norm);
			}
		}
	}
}
