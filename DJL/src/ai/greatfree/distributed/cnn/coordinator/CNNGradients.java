package ai.greatfree.distributed.cnn.coordinator;

import java.io.IOException;
import java.util.List;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;

import ai.djl.ndarray.NDManager;
import ai.greatfree.distributed.cnn.message.CNNParam;
import ai.greatfree.distributed.cnn.utils.CNNTools;

/**
 * Created: 10/16/2026
 * CNN counterpart to RNNGradients.
 */
final class CNNGradients
{
    public static double getAggregation(List<CNNParam> params, String childKey)
            throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
    {
        List<CNNParam> otherParams = Coordinator.TRAIN().getAllParamsExcept(childKey);
        otherParams.addAll(params);
        try (NDManager manager = NDManager.newBaseManager())
        {
            return CNNTools.getNorm(manager, otherParams);
        }
    }
}
