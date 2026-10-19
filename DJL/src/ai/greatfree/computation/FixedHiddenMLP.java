package ai.greatfree.computation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Created: 08/19/2024, Bing Li
 */
class FixedHiddenMLP extends AbstractBlock
{
	private static final byte VERSION = 1;

	private Block hidden20;
	private NDArray constantParamWeight;
	private NDArray constantParamBias;

	public FixedHiddenMLP()
	{
		super(VERSION);
		this.hidden20 = addChildBlock("denseLayer", Linear.builder().setUnits(20).build());
	}

	@Override
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params)
	{
		NDList current = inputs;

		// Fully connected layer
		current = this.hidden20.forward(parameterStore, current, training);
		// Use the constant parameters NDArray
		// Call the NDArray internal method `linear()` to do calculation
		current = Linear.linear(current.singletonOrThrow(), this.constantParamWeight, this.constantParamBias);
		// Relu Activation
		current = new NDList(Activation.relu(current.singletonOrThrow()));
		// Reuse the fully connected layer. This is equivalent to sharing
		// parameters with two fully connected layers
		current = this.hidden20.forward(parameterStore, current, training);

		// Here in Control flow, we return the scalar
		// for comparison
		while (current.head().abs().sum().getFloat() > 1)
		{
			current.head().divi(2);
		}
		return new NDList(current.head().abs().sum());
	}

	@Override
	public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes)
	{
		Shape[] shapes = inputShapes;
		for (Block child : getChildren().values())
		{
			child.initialize(manager, dataType, shapes);
			shapes = child.getOutputShapes(shapes);
		}
		// Initialize constant parameter layer
		this.constantParamWeight = manager.randomUniform(-0.07f, 0.07f, new Shape(20, 20));
		this.constantParamBias = manager.zeros(new Shape(20));
	}

	@Override
	public Shape[] getOutputShapes(Shape[] inputs)
	{
		return new Shape[] { new Shape(1) }; // we return a scalar so the shape is 1
	}
}

