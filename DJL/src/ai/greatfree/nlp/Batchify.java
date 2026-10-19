package ai.greatfree.nlp;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/24/2024, Bing Li
 */
final class Batchify
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDList x1 = new NDList(manager.create(new int[] { 1 }), manager.create(new int[] { 2, 2 }), manager.create(new int[] { 3, 3, 3, 3 }));
			NDList x2 = new NDList(manager.create(new int[] { 1 }), manager.create(new int[] { 2, 2, 2 }), manager.create(new int[] { 3, 3 }));

			NDList batchedData = NLPUtils.batchifyData(manager, new NDList[] { x1, x2 });
			String[] names = new String[] { "centers", "contexts_negatives", "masks", "labels" };
			for (int i = 0; i < batchedData.size(); i++)
			{
				System.out.println(names[i] + " shape: " + batchedData.get(i));
			}
		}
	}
}

