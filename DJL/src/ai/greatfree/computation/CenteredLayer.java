package ai.greatfree.computation;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Created: 08/19/2024, Bing Li
 */
class CenteredLayer extends AbstractBlock
{
	@Override
	public Shape[] getOutputShapes(Shape[] inputs)
	{
		return inputs;
	}

	@Override
	protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params)
	{
		NDList current = inputs;
		// Subtract the mean from the input
		return new NDList(current.head().sub(current.head().mean()));
	}
}

