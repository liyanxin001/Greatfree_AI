package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/13/2024, Bing li
 */
final class BasicProbability
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			float[] fairProbsArr = new float[6];
			for (int i = 0; i < fairProbsArr.length; i++)
			{
				fairProbsArr[i] = 1f / 6;
			}
			NDArray fairProbs = manager.create(fairProbsArr);
//			NDArray counts = manager.randomMultinomial(1, fairProbs);
//			NDArray counts = manager.randomMultinomial(10, fairProbs);
			NDArray counts = manager.randomMultinomial(1000, fairProbs);
			System.out.println(counts);
			System.out.println(counts.div(1000));
		}
	}
}
