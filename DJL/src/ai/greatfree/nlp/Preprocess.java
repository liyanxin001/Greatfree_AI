package ai.greatfree.nlp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.util.Pair;
import ai.greatfree.utils.Vocab;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

/**
 * Created: 08/24/2024, Bing Li
 */
final class Preprocess
{
	public static void main(String[] args) throws IOException
	{
		Scanner scan = new Scanner(System.in);
		String home = "/Users/libing/Temp/";
//		String home = "/home/greatfree/Temp/";
//		String home = "/home/libing/Temp/";
		String[][] sentences = NLPUtils.readPTB(home, "ptb/", "ptb.train.txt");
		System.out.println("# sentences: " + sentences.length);

		/*
		 * The more frequent a word emerges, the less important it is. That is the assumption for the preprocessing. 08/26/2024, Bing Li
		 */
		Vocab vocab = new Vocab(sentences, 10, new String[] {});
//		Vocab vocab = new Vocab(sentences, 1, new String[] {});
//		Vocab vocab = new Vocab(sentences, 0, new String[] {});
//		Vocab vocab = new Vocab(sentences, 20, new String[] {});
//		Vocab vocab = new Vocab(sentences, 50, new String[] {});
//		Vocab vocab = new Vocab(sentences, 100, new String[] {});

		/*
		for (int i = 0; i < vocab.length(); i++)
		{
			System.out.println(i + ") " + vocab.toToken(Integer.valueOf(i)));
		}
		*/
		
		System.out.println("vocabulary length = " + vocab.length());
//		System.out.println("Press Enter to continue ...");
//		scan.nextLine();

		/*
		 * It retrieves words upon vocabulary from each line of the essay. 08/26/2024, Bing Li
		 */
		String[][] subsampled = NLPUtils.subSampling(sentences, vocab);

		/*
		for (int i = 0; i < subsampled.length; i++)
		{
			System.out.println("Subsampled " + i + " with " + subsampled[i].length + " values:");
			for (int j = 0; j < subsampled[i].length; j++)
			{
				System.out.print("(" + i + ", " + j + ") " + subsampled[i][j] + ", ");
			}
			System.out.println("\n-----------------------\n");
		}
		*/

//		System.out.println("Press Enter to continue ...");
//		scan.nextLine();
		
		double[] y1 = new double[sentences.length];
		for (int i = 0; i < sentences.length; i++)
		{
			y1[i] = sentences[i].length;
		}

		double[] y2 = new double[subsampled.length];

		for (int i = 0; i < subsampled.length; i++)
		{
			y2[i] = subsampled[i].length;
		}

		/*
		 * The figure is NOT shown. I think it is due to the Internet fire-wall that
		 * blocks something. 08/24/2024, Bing Li
		 */ 
		 HistogramTrace trace1 =
		 HistogramTrace.builder(y1).opacity(.75).name("origin").nBinsX(20).build();
		 HistogramTrace trace2 =
		 HistogramTrace.builder(y2).opacity(.75).name("subsampled").nBinsX(20).build();
		 
		 Layout layout = Layout.builder().barMode(Layout.BarMode.GROUP).showLegend(true).xAxis(Axis.builder().title("# tokens per sentence").build()).yAxis(Axis.builder().title("count").build()).build();
		 // The figure is NOT shown. I think it is due to the Internet fire-wall that blocks something. 08/24/2024, Bing Li
		 Plot.show(new Figure(layout, trace1, trace2));

		System.out.println(NLPUtils.compareCounts("the", sentences, subsampled));
		System.out.println(NLPUtils.compareCounts("join", sentences, subsampled));

		System.out.println("----------------------");

		Integer[][] corpus = new Integer[subsampled.length][];
		
		/*
		 * Convert the words into vectors. 08/26/2024, Bing Li
		 */
		for (int i = 0; i < subsampled.length; i++)
		{
			corpus[i] = vocab.getIdxs(subsampled[i]);
		}
//		for (int i = 0; i < 3; i++)
		/*
		for (int i = 0; i < 100; i++)
		{
			System.out.println(Arrays.toString(corpus[i]));
		}

		System.out.println("Press Enter to continue ...");
		scan.nextLine();
		*/

		System.out.println("----------------------");

		Integer[][] tinyDataset = new Integer[][]
		{
			IntStream
				.range(0, 7)
				.boxed()
				.collect(Collectors.toList())
				.toArray(new Integer[] {}),

				IntStream.range(7, 10)
					.boxed()
					.collect(Collectors.toList())
					.toArray(new Integer[] {})
		};

		System.out.println("tinyDataset " + Arrays.deepToString(tinyDataset));

		System.out.println("----------------------");
		Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> centerContextPair = NLPUtils.getCentersAndContext(tinyDataset, 2);
		for (int i = 0; i < centerContextPair.getValue().size(); i++)
		{
			System.out.println("Center " + centerContextPair.getKey().get(i) + " has contexts" + centerContextPair.getValue().get(i));

			System.out.println("Center = " + vocab.toToken(centerContextPair.getKey().get(i)));
			System.out.print("Context = [");
			for (Integer entry : centerContextPair.getValue().get(i))
			{
				System.out.print(vocab.toToken(entry) + " ");
			}
			System.out.print("]\n");
		}

//		System.out.println("Press Enter to continue ...");
//		scan.nextLine();

		System.out.println("----------------------");
		centerContextPair = NLPUtils.getCentersAndContext(corpus, 5);
		ArrayList<Integer> allCenters = centerContextPair.getKey();
		ArrayList<ArrayList<Integer>> allContexts = centerContextPair.getValue();
		System.out.println("# center-context pairs:" + allCenters.size());
		System.out.println("# allContexts size:" + allContexts.size());

//		for (int i = 0; i < allContexts.size(); i++)
		for (int i = 0; i < centerContextPair.getValue().size(); i++)
		{
			System.out.println("Center = " + vocab.toToken(centerContextPair.getKey().get(i)));
			System.out.print("Context = [");
//			for (Integer entry : allContexts.get(i))
			for (Integer entry : centerContextPair.getValue().get(i))
			{
				System.out.print(vocab.toToken(entry) + " ");
			}
			System.out.print("]\n");
		}
		
//		System.out.println("----------------------");
//		System.out.println("Press Enter to continue ...");
//		scan.nextLine();
		
		ArrayList<ArrayList<Integer>> allNegatives = NLPUtils.getNegatives(allContexts, corpus, 5);
//		for (ArrayList<Integer> entry : allNegatives)
		for (int i = 0; i < allNegatives.size(); i++)
		{
//			System.out.println("Center = " + vocab.toToken(centerContextPair.getKey().get(i)));
//			System.out.println("Center = " + vocab.toToken(i));
			System.out.print("Context = [");
			for (Integer entry : allNegatives.get(i))
			{
				System.out.print(vocab.toToken(entry) + " ");
			}
			System.out.print("]\n");
		}
		System.out.println("# allNegatives size:" + allNegatives.size());
		scan.close();
	}
}

