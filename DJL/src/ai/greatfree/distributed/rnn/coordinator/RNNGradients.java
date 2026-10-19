package ai.greatfree.distributed.rnn.coordinator;

import java.io.IOException;
import java.util.List;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;

import ai.djl.ndarray.NDManager;
import ai.greatfree.distributed.rnn.message.RNNParam;
import ai.greatfree.distributed.rnn.utils.RNNTools;

/**
 * Created: 09/01/2024, Bing Li
 */
final class RNNGradients
{
	public static double getAggregation(List<RNNParam> params, String childKey) throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
	{
		List<RNNParam> otherParams = Coordinator.TRAIN().getAllParamsExcept(childKey);
		otherParams.addAll(params);
		try (NDManager manager = NDManager.newBaseManager())
		{
			return RNNTools.getNorm(manager, otherParams);
		}
	}
}

