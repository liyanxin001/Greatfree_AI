package ai.greatfree.distributed.rnn.utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.greatfree.util.Builder;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/31/2024, Bing Li
 */
public class SequentialTextLoader implements Iterable<NDList>
{
	public ArrayList<NDList> dataIter;
	public List<Integer> corpus;
	public Vocab vocab;
	public int batchSize;
	public int numSteps;

	public SequentialTextLoader(SequentialTextLoaderBuilder builder) throws Exception
	{
		Pair<List<Integer>, Vocab> corpusVocabPair = RNNTools.loadCorpus(builder.getFileName()	, builder.getStartIndex(), builder.getEndIndex(), builder.getMaxTokens());
		this.corpus = corpusVocabPair.getKey();
		this.vocab = corpusVocabPair.getValue();
		
		this.batchSize = builder.getBatchSize();
		this.numSteps = builder.getNumSteps();
		if (builder.userRandomIter())
		{
			this.dataIter = RNNTools.seqDataIterRandom(this.corpus, this.batchSize, this.numSteps, builder.getManager());
		}
		else
		{
			this.dataIter = RNNTools.seqDataIterSequential(this.corpus, this.batchSize, this.numSteps, builder.getManager());
		}
	}

	public static class SequentialTextLoaderBuilder implements Builder<SequentialTextLoader>
	{
		private String fileName;
		private int startIndex;
		private int endIndex;
		private int batchSize;
		private int numSteps;
		private boolean useRandomIter;
		private int maxTokens;
		private NDManager manager;
		
		public SequentialTextLoaderBuilder()
		{
		}

		public SequentialTextLoaderBuilder fileName(String fileName)
		{
			this.fileName = fileName;
			return this;
		}

		public SequentialTextLoaderBuilder startIndex(int startIndex)
		{
			this.startIndex = startIndex;
			return this;
		}

		public SequentialTextLoaderBuilder endIndex(int endIndex)
		{
			this.endIndex = endIndex;
			return this;
		}

		public SequentialTextLoaderBuilder batchSize(int batchSize)
		{
			this.batchSize = batchSize;
			return this;
		}

		public SequentialTextLoaderBuilder numSteps(int numSteps)
		{
			this.numSteps = numSteps;
			return this;
		}

		public SequentialTextLoaderBuilder useRandomIter(boolean useRandomIter)
		{
			this.useRandomIter = useRandomIter;
			return this;
		}

		public SequentialTextLoaderBuilder maxTokens(int maxTokens)
		{
			this.maxTokens = maxTokens;
			return this;
		}

		public SequentialTextLoaderBuilder manager(NDManager manager)
		{
			this.manager = manager;
			return this;
		}

		@Override
		public SequentialTextLoader build()
		{
			try
			{
				return new SequentialTextLoader(this);
			}
			catch (Exception e)
			{
				e.printStackTrace();
				return null;
			}
		}
		
		public String getFileName()
		{
			return this.fileName;
		}
		
		public int getStartIndex()
		{
			return this.startIndex;
		}
		
		public int getEndIndex()
		{
			return this.endIndex;
		}
		
		public int getBatchSize()
		{
			return this.batchSize;
		}
		
		public int getNumSteps()
		{
			return this.numSteps;
		}
		
		public boolean userRandomIter()
		{
			return this.useRandomIter;
		}
		
		public int getMaxTokens()
		{
			return this.maxTokens;
		}
		
		public NDManager getManager()
		{
			return this.manager;
		}
	}
	
	@Override
	public Iterator<NDList> iterator()
	{
		return this.dataIter.iterator();
	}
}

