package ai.greatfree.distributed.rnn.child;

import java.io.IOException;
import java.util.Calendar;
import java.util.List;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.FileManager;

import ai.greatfree.distributed.rnn.child.app.ChildProfile;
import ai.greatfree.distributed.rnn.child.app.RNNTrainer;
import ai.greatfree.distributed.rnn.message.CollectGradientsRequest;
import ai.greatfree.distributed.rnn.message.CollectGradientsResponse;
import ai.greatfree.distributed.rnn.message.DistributeTrainingDataRequest;
import ai.greatfree.distributed.rnn.message.DistributeTrainingDataResponse;
import ai.greatfree.distributed.rnn.message.InitRequest;
import ai.greatfree.distributed.rnn.message.InitResponse;
import ai.greatfree.distributed.rnn.message.PredictRequest;
import ai.greatfree.distributed.rnn.message.PredictResponse;
import ai.greatfree.distributed.rnn.message.RNNAppID;
import ai.greatfree.distributed.rnn.utils.RNNTools;
import edu.greatfree.cluster.child.ChildTask;
import edu.greatfree.cluster.message.ClusterNotification;
import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.ClusterResponse;
import edu.greatfree.cluster.message.InterChildrenNotification;
import edu.greatfree.cluster.message.InterChildrenRequest;
import edu.greatfree.cluster.message.InterStaticNotification;
import edu.greatfree.cluster.message.InterStaticRequest;
import edu.greatfree.cluster.message.IntercastNotification;
import edu.greatfree.cluster.message.IntercastRequest;
import edu.greatfree.cluster.message.IsReplicaAvailableRequest;
import edu.greatfree.cluster.message.IsReplicaAvailableResponse;
import edu.greatfree.cluster.message.Replica;
import edu.greatfree.cluster.message.ReplicationRequest;
import edu.greatfree.cluster.message.ReplicationResponse;
import edu.greatfree.framework.cluster.multicast.message.ClusterAppID;
import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
final class RNNChildTask extends ChildTask
{
	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.child");

	@Override
	public InterStaticNotification assignDestinations(InterStaticNotification notification)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public InterStaticRequest assignDestinations(InterStaticRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IntercastNotification assignDestinations(IntercastNotification notification)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IntercastRequest assignDestinations(IntercastRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IsReplicaAvailableResponse isReplicaAvailable(IsReplicaAvailableRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public InterChildrenNotification prepareNotification(IntercastNotification notification)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Replica prepareReplication(int arg0)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public InterChildrenRequest prepareRequest(IntercastRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void processFailedReplication(ReplicationResponse response)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void processNotification(ClusterNotification notification)
	{
		switch (notification.getAppID())
		{
			case RNNAppID.START_TRAINING_NOTIFICATION:
				log.info("START_TRAINING_NOTIFICATION received @" + Calendar.getInstance().getTime());
				try
				{
					RNNTrainer.CHILD().train();
				}
				catch (ClassNotFoundException | RemoteReadException | IOException | InterruptedException e)
				{
					e.printStackTrace();
				}
				break;

			case ClusterAppID.SHUTDOWN_CHILDREN_NOTIFICATION:
				log.info("SHUTDOWN_CHILDREN_NOTIFICATION received @" + Calendar.getInstance().getTime());
				try
				{
					RNNChild.CLUSTER().stop();
				}
				catch (ClassNotFoundException | IOException | InterruptedException | RemoteReadException e)
				{
					e.printStackTrace();
				}
				break;
		}
	}

	@Override
	public void processNotification(InterChildrenNotification notification)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void processNotification(InterChildrenNotification notification, List<String> distinationKeys)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public MulticastResponse processRequest(ClusterRequest request)
	{
		switch (request.getAppID())
		{
			case RNNAppID.DISTRIBUTE_TRAINING_DATA_REQUEST:
				log.info("DISTRIBUTE_TRAINING_DATA_REQUEST received @" + Calendar.getInstance().getTime());
				DistributeTrainingDataRequest dtdr = (DistributeTrainingDataRequest)request;
				try
				{
					if (!FileManager.isDirExisted(ChildProfile.CHILD().getTrainingDataPath()))
					{
						FileManager.makeDir(ChildProfile.CHILD().getTrainingDataPath());
					}
					ChildProfile.CHILD().setTrainingDataFile(dtdr.getFileName());
//					FileManager.writeLines(ChildProfile.CHILD().getTrainingDataPath() + dtdr.getFileName(), dtdr.getTexts());
					FileManager.writeLines(ChildProfile.CHILD().getTrainingDataFile(), dtdr.getTexts());
				}
				catch (IOException e)
				{
					return new DistributeTrainingDataResponse(false, dtdr.getChildKey(), dtdr.getCollaboratorKey());
				}
				return new DistributeTrainingDataResponse(true, dtdr.getChildKey(), dtdr.getCollaboratorKey());
				
			case RNNAppID.INIT_REQUEST:
				log.info("INIT_REQUEST received @" + Calendar.getInstance().getTime());
				InitRequest ir = (InitRequest)request;
				try
				{
					RNNTrainer.CHILD().init();
				}
				catch (Exception e)
				{
					e.printStackTrace();
					return new InitResponse(RNNChild.CLUSTER().getChildKey(), false, ir.getCollaboratorKey());
				}
				return new InitResponse(RNNChild.CLUSTER().getChildKey(), true, ir.getCollaboratorKey());
				
			case RNNAppID.COLLECT_GRADIENTS_REQUEST:
				log.info("COLLECT_GRADIENTS_REQUEST received @" + Calendar.getInstance().getTime());
				CollectGradientsRequest cgr = (CollectGradientsRequest)request;
				return new CollectGradientsResponse(RNNTools.getGradients(RNNTrainer.CHILD().getRNN()), cgr.getCollaboratorKey());
				
			case RNNAppID.PREDICT_REQUEST:
				log.info("PREDICT_REQUEST received @" + Calendar.getInstance().getTime());
				PredictRequest pr = (PredictRequest)request;
				return new PredictResponse(RNNTrainer.CHILD().predict(pr.getPrefix()), pr.getCollaboratorKey());
		}
		return null;
	}

	@Override
	public List<MulticastResponse> processRequest(InterChildrenRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<MulticastResponse> processRequest(InterChildrenRequest request, List<String> distinationKeys)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void processResponse(ClusterResponse response)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public ReplicationResponse replicate(ReplicationRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}

}
