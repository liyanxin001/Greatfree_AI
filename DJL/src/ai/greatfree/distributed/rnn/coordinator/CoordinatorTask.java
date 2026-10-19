package ai.greatfree.distributed.rnn.coordinator;

import java.io.IOException;
import java.util.Calendar;
import java.util.logging.Logger;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.exceptions.ThreadNotExistedException;
import org.greatfree.message.ServerMessage;

import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.message.GradientsAggregationRequest;
import ai.greatfree.distributed.rnn.message.GradientsAggregationResponse;
import ai.greatfree.distributed.rnn.message.RNNAppID;
import ai.greatfree.distributed.rnn.message.TrainingDoneNotification;
import edu.greatfree.container.message.Notification;
import edu.greatfree.container.message.Request;
import edu.greatfree.container.server.ServerTask;

/**
 * Created: 09/01/2024, Bing Li
 */
final class CoordinatorTask extends ServerTask
{
	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.coordinator");

	@Override
	public ServerMessage processRequest(Request request)
	{
		switch (request.getAppID())
		{
			case RNNAppID.GRADIENTS_AGGREGATION_REQUEST:
				log.info("GRADIENTS_AGGREGATION_REQUEST received @" + Calendar.getInstance().getTime());
				GradientsAggregationRequest gar = (GradientsAggregationRequest)request;
				try
				{
					return new GradientsAggregationResponse(RNNGradients.getAggregation(gar.getParams(), gar.getChildKey()), true);
				}
				catch (ClassNotFoundException | RemoteReadException | IOException | NullClassConversionException e)
				{
					return new GradientsAggregationResponse(RNNConfig.NO_NORM, false);
				}
		}
		return null;
	}

	@Override
	public void processNotification(Notification notification) throws ThreadNotExistedException
	{
		switch (notification.getAppID())
		{
			case RNNAppID.TRAINING_DONE_NOTIFICATION:
				log.info("TRAINING_DONE_NOTIFICATION received @" + Calendar.getInstance().getTime());
				TrainingDoneNotification tdn = (TrainingDoneNotification)notification;
				if (!Coordinator.TRAIN().isDone(tdn.getChildKey()))
				{
					log.info(Coordinator.TRAIN().getDoneCount() + " children have finished their training!");
				}
				else
				{
					log.info("All the (" + Coordinator.TRAIN().getDoneCount() + ") children have finished their training! You can start to predict now!");
				}
				break;
		}
	}
}

