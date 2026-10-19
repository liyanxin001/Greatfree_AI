package ai.greatfree.distributed.rnn.utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.FileManager;
import org.greatfree.util.UtilConfig;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Pair;
import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.child.RNNChild;
import ai.greatfree.distributed.rnn.child.app.ChildProfile;
import ai.greatfree.distributed.rnn.message.RNNParam;
import ai.greatfree.utils.Accumulator;
import ai.greatfree.utils.RNNModelScratch;
import ai.greatfree.utils.StopWatch;
import ai.greatfree.utils.Training;
import ai.greatfree.utils.Vocab;
import ai.greatfree.utils.Functions.SimpleFunction;
import ai.greatfree.utils.Functions.voidTwoFunction;

/**
 * Created: 
 */
public final class RNNTools
{
	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.utils");

	/**
	 * Return the iterator and the vocabulary of the time machine dataset.
	 */
//	public static Pair<ArrayList<NDList>, Vocab> loadTrainingData(int batchSize, int numSteps, boolean useRandomIter, int maxTokens, NDManager manager) throws IOException, Exception
	@SuppressWarnings("null")
	public static Pair<ArrayList<NDList>, Vocab> loadTrainingData(SequentialTextLoader loader) throws IOException, Exception
	{
		if (loader == null)
		{
			log.info("2) loader is NULL!!!");
		}
		if (loader.dataIter == null)
		{
			log.info("loader.dataIter is NULL!!!");
		}
		
		if (loader.vocab == null)
		{
			log.info("loader.vocab is NULL!!!");
		}
//		SequentialTextLoader seqData = new SequentialTextLoader(batchSize, numSteps, useRandomIter, maxTokens, manager);
		return new Pair<ArrayList<NDList>, Vocab>(loader.dataIter, loader.vocab); // ArrayList<NDList>, Vocab
	}

	private static String[] readTexts(String textFile, int startIndex, int endIndex)
	{
		log.info("textFile = " + textFile);
		String[] lines;
		try
		{
			lines = FileManager.readLineArrayByIndex(textFile, startIndex, endIndex);
			log.info("1) lines = " + lines.length);
			for (int i = 0; i < lines.length; i++)
			{
//				log.info(lines[i]);
				lines[i] = lines[i].replaceAll(RNNConfig.REGEXPRESS_FILTER, UtilConfig.SPACE).strip().toLowerCase();
			}
			log.info("2) lines = " + lines.length);
			return lines;
		}
		catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	private static String[][] tokenize(String[] lines, String token) throws Exception
	{
		// Split text lines into word or character tokens.
		String[][] output = new String[lines.length][];
		if (token == RNNConfig.TOKENIZE_BY_WORD)
		{
			for (int i = 0; i < output.length; i++)
			{
				output[i] = lines[i].split(UtilConfig.SPACE);
			}
		}
		else if (token == RNNConfig.TOKENIZE_BY_CHAR)
		{
			for (int i = 0; i < output.length; i++)
			{
				output[i] = lines[i].split(UtilConfig.EMPTY_STRING);
			}
		}
		else
		{
			throw new Exception(RNNConfig.ERROR_TOKEN + token);
		}
		return output;
	}

	public static Pair<List<Integer>, Vocab> loadCorpus(String fileName, int startIndex, int endIndex, int maxTokens) throws Exception
	{
		log.info("Training data is loading ...");
		log.info("fileName = " + fileName);
		String[] lines = readTexts(fileName, startIndex, endIndex);
		log.info("Training data is loaded ...");
		
		log.info("lines length = " + lines.length);

		int length = lines.length;
		for (int i = 0; i < length; i++)
		{
			System.out.println(i + ") " + lines[i]);
		}
		
		String[][] tokens = tokenize(lines, RNNConfig.TOKENIZE_BY_CHAR);
		Vocab vocab = new Vocab(tokens, 0, new String[0]);
		// Since each text line in the time machine dataset is not necessarily a
		// sentence or a paragraph, flatten all the text lines into a single list
		List<Integer> corpus = new ArrayList<>();
		for (String[] token : tokens)
		{
			for (String s : token)
			{
				if (!s.isEmpty())
				{
					corpus.add(vocab.getIdx(s));
				}
			}
		}
		if (maxTokens > 0)
		{
			corpus = corpus.subList(0, maxTokens);
		}
		return new Pair<>(corpus, vocab);
	}

	private static NDArray normal(Shape shape, Device device, NDManager manager)
	{
		return manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device);
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

	private static ArrayList<Integer> data(int pos, List<Integer> corpus, int numSteps)
	{
		// Return a sequence of length `numSteps` starting from `pos`
		return new ArrayList<Integer>(corpus.subList(pos, pos + numSteps));
	}

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

	/** Generate new characters following the `prefix`. */
	public static String predict(String prefix, int numPreds, RNNModelScratch net, Vocab vocab, Device device, NDManager manager)
	{
		NDList state = net.beginState(1, device);
		List<Integer> outputs = new ArrayList<>();
		outputs.add(vocab.getIdx(UtilConfig.EMPTY_STRING + prefix.charAt(0)));
		SimpleFunction<NDArray> getInput = () -> manager.create(outputs.get(outputs.size() - 1)).toDevice(device, false).reshape(new Shape(1, 1));
		for (char c : prefix.substring(1).toCharArray())
		{ // Warm-up period
			state = (NDList) net.forward(getInput.apply(), state).getValue();
			outputs.add(vocab.getIdx(UtilConfig.EMPTY_STRING + c));
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

	private static void associateWithMan(RNNModelScratch net, NDManager manager)
	{
		for (NDArray p : net.params)
		{
			NDArray gradient = p.getGradient();
			gradient.attach(manager);
		}
	}
	
	public static void gradClippingLocally(RNNModelScratch net, int theta)
	{
		double result = 0;
		NDArray gradient;
		for (NDArray param : net.params)
		{
//			log.info("param bytes size = " + p.encode().length);
			gradient = param.getGradient();
//			log.info("gradient bytes size = " + gradient.encode().length);
			
			/**
			 * The method, associateWithMan, is responsible for the below task. 08/29/2024, Bing Li
			 * 
			 * The below line is NOT necessary? 08/29/2024, Bing Li
			 */
//			gradient.attach(manager);
			result += gradient.pow(RNNConfig.SQUARE).sum().getFloat();
		}
		double norm = Math.sqrt(result);
		if (norm > theta)
		{
			for (NDArray param : net.params)
			{
				gradient = param.getGradient();
				gradient.muli(theta / norm);
			}
		}
	}

	public static double getNorm(NDManager manager, List<RNNParam> params)
	{
		double result = 0;
		NDArray gradient;
		for (RNNParam param : params)
		{
			gradient = NDArray.decode(manager, param.getParams());
			result += gradient.pow(RNNConfig.SQUARE).sum().getFloat();
		}
		return Math.sqrt(result);
	}
	
	public static List<RNNParam> getGradients(RNNModelScratch net)
	{
		List<RNNParam> params = new ArrayList<RNNParam>();
		for (NDArray param : net.params)
		{
			params.add(new RNNParam(param.encode()));
		}
		return params;
	}

	public static void gradClippingGlobally(RNNModelScratch net, int theta) throws ClassNotFoundException, RemoteReadException, IOException
	{
		List<RNNParam> params = new ArrayList<RNNParam>();
		for (NDArray param : net.params)
		{
			params.add(new RNNParam(param.encode()));
		}
		double norm = RNNChild.CLUSTER().getGradientsAggregation(params);
		if (norm > theta)
		{
			NDArray gradient;
			for (NDArray param : net.params)
			{
				gradient = param.getGradient();
				gradient.muli(theta / norm);
			}
		}
	}

	private static Pair<Double, Double> trainByEpoch(RNNModelScratch net, List<NDList> trainIter, Loss loss, voidTwoFunction<Integer, NDManager> updater, Device device, boolean useRandomIter, NDManager manager) throws ClassNotFoundException, RemoteReadException, IOException
	{
		StopWatch watch = new StopWatch();
		watch.start();
		Accumulator metric = new Accumulator(RNNConfig.ACCUMULATOR_NUM); // Sum of training loss, no. of tokens
		try (NDManager childManager = manager.newSubManager())
		{
			NDList state = null;
			NDArray X;
			NDArray Y;
			NDArray y;
			Pair<NDArray, NDList> pairResult;
			NDArray yHat;
			NDArray l;
			for (NDList pair : trainIter)
			{
				X = pair.get(0).toDevice(device, true);
				X.attach(childManager);
				Y = pair.get(1).toDevice(device, true);
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
				
				y = Y.transpose().reshape(new Shape(-1));
				X = X.toDevice(device, false);
				y = y.toDevice(device, false);
				try (GradientCollector gc = manager.getEngine().newGradientCollector())
				{
					pairResult = net.forward(X, state);
					yHat = pairResult.getKey();
					state = pairResult.getValue();
					l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
					gc.backward(l);
					metric.add(new float[] { l.getFloat() * y.size(), y.size() });
				}
				associateWithMan(net, childManager);
				if (!ChildProfile.CHILD().isLocalAggregationFinished(RNNConfig.LOCAL_AGGREGATION_TIME))
				{
					gradClippingLocally(net, RNNConfig.THETA);
				}
				else
				{
					ChildProfile.CHILD().clearLocalAggregationTimes();
					gradClippingGlobally(net, RNNConfig.THETA);
				}
				updater.apply(RNNConfig.UPDATE_BATCH_SIZE, childManager); // Since the `mean` function has been invoked
			}
		}
		return new Pair<>(Math.exp(metric.get(UtilConfig.ZERO) / metric.get(UtilConfig.ONE)), metric.get(UtilConfig.ONE) / watch.stop());
	}

	public static Function<String, String> train(RNNModelScratch net, List<NDList> trainIter, Vocab vocab, int lr, int numEpochs, Device device, boolean useRandomIter, NDManager manager) throws ClassNotFoundException, RemoteReadException, IOException
	{
		SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
		voidTwoFunction<Integer, NDManager> updater = (batchSize, subManager) -> Training.sgd(net.params, lr, batchSize, subManager);
		Function<String, String> predictor = (prefix) -> predict(prefix, RNNConfig.NUM_PERIODS, net, vocab, device, manager);
		double ppl = 0.0;
		double speed = 0.0;
		Pair<Double, Double> pair;
		for (int epoch = 0; epoch < numEpochs; epoch++)
		{
			pair = trainByEpoch(net, trainIter, loss, updater, device, useRandomIter, manager);
			ppl = pair.getKey();
			speed = pair.getValue();
			/*
			if ((epoch + UtilConfig.ONE) % 10 == 0)
			{
				log.info("Epoch = " + (epoch + UtilConfig.ONE) + ", PPL = " + ppl);
			}
			*/
			System.out.println("----------------------------------------------------------------------------");
			System.out.println("Epoch = " + (epoch + UtilConfig.ONE) + ", PPL = " + ppl);
			System.out.println("----------------------------------------------------------------------------");
		}
		System.out.format("perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
		return predictor;
	}
}
