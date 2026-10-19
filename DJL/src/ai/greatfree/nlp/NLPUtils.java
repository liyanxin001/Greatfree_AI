package ai.greatfree.nlp;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.ZipUtils;
import ai.greatfree.utils.RandomGenerator;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/24/2024, Bing li
 */
final class NLPUtils
{
	public static String[][] readPTB(String homePath, String txtDir, String txtFile) throws IOException
	{
		String ptbURL = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip";
		@SuppressWarnings("deprecation")
		InputStream input = new URL(ptbURL).openStream();
		ZipUtils.unzip(input, Paths.get(homePath));

		ArrayList<String> lines = new ArrayList<>();
		// File file = new File("./ptb/ptb.train.txt");
		File file = new File(homePath + txtDir + txtFile);
		Scanner myReader = new Scanner(file);
		String line;
//		int index = 0;
		while (myReader.hasNextLine())
		{
			line = myReader.nextLine();
//			System.out.println(index++ + ") " + line);
			lines.add(line);
		}
		String[][] tokens = new String[lines.size()][];
		for (int i = 0; i < lines.size(); i++)
		{
			tokens[i] = lines.get(i).trim().split(" ");
		}
		myReader.close();
		return tokens;
	}

	public static boolean keep(String token, LinkedHashMap<?, Integer> counter, int numTokens)
	{
		// Return True if to keep this token during subsampling
		return new Random().nextFloat() < Math.sqrt(1e-4 / counter.get(token) * numTokens);
	}

	public static String[][] subSampling(String[][] sentences, Vocab vocab)
	{
		for (int i = 0; i < sentences.length; i++)
		{
			for (int j = 0; j < sentences[i].length; j++)
			{
				sentences[i][j] = vocab.idxToToken.get(vocab.getIdx(sentences[i][j]));
			}
		}
		// Count the frequency for each word
		LinkedHashMap<?, Integer> counter = Vocab.countCorpus2D(sentences);
		int numTokens = 0;
		for (Integer value : counter.values())
		{
			numTokens += value;
		}

		// Now do the subsampling
		String[][] output = new String[sentences.length][];
		for (int i = 0; i < sentences.length; i++)
		{
			ArrayList<String> tks = new ArrayList<>();
			for (int j = 0; j < sentences[i].length; j++)
			{
				String tk = sentences[i][j];
				if (keep(sentences[i][j], counter, numTokens))
				{
					tks.add(tk);
				}
			}
			output[i] = tks.toArray(new String[tks.size()]);
		}
		return output;
	}

	public static String compareCounts(String token, String[][] sentences, String[][] subsampled)
	{
		int beforeCount = 0;
		for (int i = 0; i < sentences.length; i++)
		{
			for (int j = 0; j < sentences[i].length; j++)
			{
				if (sentences[i][j].equals(token))
					beforeCount += 1;
			}
		}

		int afterCount = 0;
		for (int i = 0; i < subsampled.length; i++)
		{
			for (int j = 0; j < subsampled[i].length; j++)
			{
				if (subsampled[i][j].equals(token))
					afterCount += 1;
			}
		}
		return "# of \"" + token + "\": before=" + beforeCount + ", after=" + afterCount;
	}

//	private final static Logger log = Logger.getLogger("io.greatfree.nlp");

	public static Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> getCentersAndContext(Integer[][] corpus, int maxWindowSize)
	{
		ArrayList<Integer> centers = new ArrayList<>();
		ArrayList<ArrayList<Integer>> contexts = new ArrayList<>();
		for (Integer[] line : corpus)
		{
			// Each sentence needs at least 2 words to form a "central target word
			// - context word" pair
			if (line.length < 2)
			{
				continue;
			}
			centers.addAll(Arrays.asList(line));
			for (int i = 0; i < line.length; i++)
			{ // Context window centered at i
				int windowSize = new Random().nextInt(maxWindowSize - 1) + 1;
				List<Integer> indices = IntStream.range(Math.max(0, i - windowSize), Math.min(line.length, i + 1 + windowSize)).boxed().collect(Collectors.toList());
				// Exclude the central target word from the context words
				indices.remove(indices.indexOf(i));
				ArrayList<Integer> context = new ArrayList<>();
				for (Integer idx : indices)
				{
					context.add(line[idx]);
				}
				contexts.add(context);
			}
		}
		return new Pair<>(centers, contexts);
	}

	public static ArrayList<ArrayList<Integer>> getNegatives(ArrayList<ArrayList<Integer>> allContexts, Integer[][] corpus, int K)
	{
		LinkedHashMap<?, Integer> counter = Vocab.countCorpus2D(corpus);
		ArrayList<Double> samplingWeights = new ArrayList<>();
		for (Map.Entry<?, Integer> entry : counter.entrySet())
		{
			samplingWeights.add(Math.pow(entry.getValue(), .75));
		}
		ArrayList<ArrayList<Integer>> allNegatives = new ArrayList<>();
		RandomGenerator generator = new RandomGenerator(samplingWeights);
		for (ArrayList<Integer> contexts : allContexts)
		{
			ArrayList<Integer> negatives = new ArrayList<>();
			while (negatives.size() < contexts.size() * K)
			{
				Integer neg = generator.draw();
				// Noise words cannot be context words
				if (!contexts.contains(neg))
				{
					negatives.add(neg);
				}
			}
			allNegatives.add(negatives);
		}
		return allNegatives;
	}

	public static NDList batchifyData(NDManager manager, NDList[] data)
	{
		NDList centers = new NDList();
		NDList contextsNegatives = new NDList();
		NDList masks = new NDList();
		NDList labels = new NDList();

		long maxLen = 0;
		for (NDList ndList : data)
		{ // center, context, negative = ndList
			maxLen = Math.max(maxLen, ndList.get(1).countNonzero().getLong() + ndList.get(2).countNonzero().getLong());
		}
		for (NDList ndList : data)
		{ // center, context, negative = ndList
			NDArray center = ndList.get(0);
			NDArray context = ndList.get(1);
			NDArray negative = ndList.get(2);

			int count = 0;
			for (int i = 0; i < context.size(); i++)
			{
				// If a 0 is found, we want to stop adding these
				// values to NDArray
				if (context.get(i).getInt() == 0)
				{
					break;
				}
				contextsNegatives.add(context.get(i).reshape(1));
				masks.add(manager.create(1).reshape(1));
				labels.add(manager.create(1).reshape(1));
				count += 1;
			}
			for (int i = 0; i < negative.size(); i++)
			{
				// If a 0 is found, we want to stop adding these
				// values to NDArray
				if (negative.get(i).getInt() == 0)
				{
					break;
				}
				contextsNegatives.add(negative.get(i).reshape(1));
				masks.add(manager.create(1).reshape(1));
				labels.add(manager.create(0).reshape(1));
				count += 1;
			}
			// Fill with zeroes remaining array
			while (count != maxLen)
			{
				contextsNegatives.add(manager.create(0).reshape(1));
				masks.add(manager.create(0).reshape(1));
				labels.add(manager.create(0).reshape(1));
				count += 1;
			}

			// Add this NDArrays to output NDArrays
			centers.add(center.reshape(1));
		}
		return new NDList(NDArrays.concat(centers).reshape(data.length, -1), NDArrays.concat(contextsNegatives).reshape(data.length, -1), NDArrays.concat(masks).reshape(data.length, -1), NDArrays.concat(labels).reshape(data.length, -1));
	}

	@SuppressWarnings("unchecked")
	public static NDList convertNDArray(Object[] data, NDManager manager)
	{
		ArrayList<Integer> centers = (ArrayList<Integer>) data[0];
		ArrayList<ArrayList<Integer>> contexts = (ArrayList<ArrayList<Integer>>) data[1];
		ArrayList<ArrayList<Integer>> negatives = (ArrayList<ArrayList<Integer>>) data[2];

		// Create centers NDArray
		NDArray centersNDArray = manager.create(centers.stream().mapToInt(i -> i).toArray());

		// Create contexts NDArray
		int maxLen = 0;
		for (ArrayList<Integer> context : contexts)
		{
			maxLen = Math.max(maxLen, context.size());
		}
		// Fill arrays with 0s to all have same lengths and be able to create NDArray
		for (ArrayList<Integer> context : contexts)
		{
			while (context.size() != maxLen)
			{
				context.add(0);
			}
		}
		NDArray contextsNDArray = manager.create(contexts.stream().map(u -> u.stream().mapToInt(i -> i).toArray()).toArray(int[][]::new));

		// Create negatives NDArray
		maxLen = 0;
		for (ArrayList<Integer> negative : negatives)
		{
			maxLen = Math.max(maxLen, negative.size());
		}
		// Fill arrays with 0s to all have same lengths and be able to create NDArray
		for (ArrayList<Integer> negative : negatives)
		{
			while (negative.size() != maxLen)
			{
				negative.add(0);
			}
		}
		NDArray negativesNDArray = manager.create(negatives.stream().map(u -> u.stream().mapToInt(i -> i).toArray()).toArray(int[][]::new));

		return new NDList(centersNDArray, contextsNDArray, negativesNDArray);
	}

	public static Pair<ArrayDataset, Vocab> loadDataPTB(String homePath, String txtDir, String txtFile, int batchSize, int maxWindowSize, int numNoiseWords, NDManager manager) throws IOException, TranslateException
	{
		String[][] sentences = readPTB(homePath, txtDir, txtFile);
		Vocab vocab = new Vocab(sentences, 10, new String[] {});
		String[][] subSampled = subSampling(sentences, vocab);
		Integer[][] corpus = new Integer[subSampled.length][];
		for (int i = 0; i < subSampled.length; i++)
		{
			corpus[i] = vocab.getIdxs(subSampled[i]);
		}
		Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> pair = getCentersAndContext(corpus, maxWindowSize);
		ArrayList<ArrayList<Integer>> negatives = getNegatives(pair.getValue(), corpus, numNoiseWords);

		NDList ndArrays = convertNDArray(new Object[] { pair.getKey(), pair.getValue(), negatives }, manager);
		ArrayDataset dataset = new ArrayDataset.Builder().setData(ndArrays.get(0), ndArrays.get(1), ndArrays.get(2)).optDataBatchifier(new Batchifier()
		{
			private static final long serialVersionUID = 1L;

			@Override
			public NDList batchify(NDList[] ndLists)
			{
				return batchifyData(manager, ndLists);
			}

			@Override
			public NDList[] unbatchify(NDList ndList)
			{
				return new NDList[0];
			}
		}).setSampling(batchSize, true).build();
		return new Pair<>(dataset, vocab);
	}

	public static Pair<String, String> getMaxFreqPair(HashMap<String, Integer> tokenFreqs)
	{
		HashMap<Pair<String, String>, Integer> pairs = new HashMap<>();
		for (Map.Entry<String, Integer> e : tokenFreqs.entrySet())
		{
			// Key of 'pairs' is a tuple of two consecutive symbols
			String token = e.getKey();
			Integer freq = e.getValue();
			String[] symbols = token.split(" ");
			for (int i = 0; i < symbols.length - 1; i++)
			{
				pairs.put(new Pair<>(symbols[i], symbols[i + 1]), pairs.getOrDefault(new Pair<>(symbols[i], symbols[i + 1]), 0) + freq);
			}
		}
		int max = 0; // Key of `pairs` with the max value
		Pair<String, String> maxFreqPair = null;
		for (Map.Entry<Pair<String, String>, Integer> pair : pairs.entrySet())
		{
			if (max < pair.getValue())
			{
				max = pair.getValue();
				maxFreqPair = pair.getKey();
			}
		}
		return maxFreqPair;
	}

	public static Pair<HashMap<String, Integer>, String[]> mergeSymbols(Pair<String, String> maxFreqPair, HashMap<String, Integer> tokenFreqs)
	{
		ArrayList<String> symbols = new ArrayList<>();
		symbols.add(maxFreqPair.getKey() + maxFreqPair.getValue());

		HashMap<String, Integer> newTokenFreqs = new HashMap<>();
		for (Map.Entry<String, Integer> e : tokenFreqs.entrySet())
		{
			String token = e.getKey();
			String newToken = token.replace(maxFreqPair.getKey() + " " + maxFreqPair.getValue(), maxFreqPair.getKey() + "" + maxFreqPair.getValue());
			newTokenFreqs.put(newToken, tokenFreqs.get(token));
		}
		return new Pair<HashMap<String, Integer>, String[]>(newTokenFreqs, symbols.toArray(new String[symbols.size()]));
	}

	public static List<String> segmentBPE(String[] tokens, String[] symbols)
	{
		List<String> outputs = new ArrayList<>();
		for (String token : tokens)
		{
			int start = 0;
			int end = token.length();
			ArrayList<String> curOutput = new ArrayList<>();
			// Segment token with the longest possible subwords from symbols
			while (start < token.length() && start < end)
			{
				if (Arrays.asList(symbols).contains(token.substring(start, end)))
				{
					curOutput.add(token.substring(start, end));
					start = end;
					end = token.length();
				}
				else
				{
					end -= 1;
				}
			}
			if (start < tokens.length)
			{
				curOutput.add("[UNK]");
			}
			String temp = "";
			for (String s : curOutput)
			{
				temp += s + " ";
			}
			outputs.add(temp.trim());
		}
		return outputs;
	}
}

