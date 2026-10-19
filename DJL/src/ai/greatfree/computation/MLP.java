package ai.greatfree.computation;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Created: 08/18/2024, Bing Li
 */
class MLP extends AbstractBlock
{
	private static final byte VERSION = 1;

	private Block flattenInput;
	private Block hidden256;
	private Block output10;

	// Declare a layer with model parameters. Here, we declare two fully
	// connected layers
	public MLP(int inputSize)
	{
		super(VERSION); // Dont need to worry about this

		this.flattenInput = super.addChildBlock("flattenInput", Blocks.batchFlattenBlock(inputSize));
		this.hidden256 = super.addChildBlock("hidden256", Linear.builder().setUnits(256).build());// Hidden Layer
		this.output10 = super.addChildBlock("output10", Linear.builder().setUnits(10).build()); // Output Layer
	}

	@Override
	// Define the forward computation of the model, that is, how to return
	// the required model output based on the input x
	protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params)
	{
		NDList current = inputs;
		current = this.flattenInput.forward(parameterStore, current, training);
		current = this.hidden256.forward(parameterStore, current, training);
		// We use the Activation.relu() function here
		// Since it takes in an NDArray, we call `singletonOrThrow()`
		// on the NDList `current` to get the NDArray and then
		// wrap it in a new NDList to be passed
		// to the next `forward()` call
		current = new NDList(Activation.relu(current.singletonOrThrow()));
		current = this.output10.forward(parameterStore, current, training);
		return current;
	}

	@Override
	public Shape[] getOutputShapes(Shape[] inputs)
	{
		Shape[] current = inputs;
		for (Block block : this.children.values())
		{
			current = block.getOutputShapes(current);
		}
		return current;
	}

	@Override
	public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes)
	{
//		this.hidden256.initialize(manager, dataType, new Shape(1, inputSize));
		// The line is revised by me. 08/18/2024, Bing Li
		this.hidden256.initialize(manager, dataType, new Shape(1, inputShapes.length));
		this.output10.initialize(manager, dataType, new Shape(1, 256));
	}
}

