package ai.greatfree.distributed.cnn.child;

import java.io.IOException;
import java.util.List;
import java.util.logging.Logger;





import org.greatfree.exceptions.RemoteReadException;

import org.greatfree.util.IPAddress;
import org.greatfree.util.TerminateSignal;

import ai.greatfree.distributed.cnn.CNNConfig;
import ai.greatfree.distributed.cnn.child.app.CNNTrainer;
import ai.greatfree.distributed.cnn.child.app.ChildProfile;

import ai.greatfree.distributed.rnn.message.ChildPath;
import ai.greatfree.distributed.rnn.message.ChildPathsRequest;
import ai.greatfree.distributed.rnn.message.ChildPathsResponse;
import ai.greatfree.distributed.cnn.message.GradientsAggregationRequest;
import ai.greatfree.distributed.cnn.message.GradientsAggregationResponse;
import ai.greatfree.distributed.cnn.message.CNNParam;
import ai.greatfree.distributed.rnn.message.TrainingDoneNotification;
import edu.greatfree.cluster.ClusterSpec;
import edu.greatfree.cluster.child.Child;
import edu.greatfree.cluster.child.ChildTask;
import edu.greatfree.exceptions.RootOfflineException;
import edu.greatfree.exceptions.TaskAlreadyExistedException;


/**
 * CNN counterpart to RNNChild. Singleton wrapper around the GreatFree Child
 * node, exposing cluster operations used by CNNTrainer.
 */
public final class CNNChild
{
    private final static Logger log = Logger.getLogger("ai.greatfree.distributed.cnn.child");

    private Child child;

    private CNNChild() {}

    private static CNNChild instance = new CNNChild();

    public static CNNChild CLUSTER()
    {
        return instance;   // the null check in RNNChild is redundant
    }

    // ============================================================
    //  LIFECYCLE
    // ============================================================

    public void start(ClusterSpec spec, String coordinatorName, ChildTask... tasks)
            throws ClassNotFoundException, IOException, RemoteReadException,
                   TaskAlreadyExistedException, InterruptedException
    {
        this.child = new Child(spec, tasks);
        try
        {
            this.child.start();
        }
        catch (RootOfflineException e)
        {
            log.info("Root is offline!");
        }
        log.info("Child's address is " + this.child.getAddress());

        // Ask the root for the data paths it assigned us.
        ChildPath paths = ((ChildPathsResponse) this.child.readRoot(
                new ChildPathsRequest(CNNConfig.TRAINING_DATA_PATH,
                                      CNNConfig.TRAINED_DATA_PATH))).getPath();

        ChildProfile.CHILD().setTrainingDataPath(paths.getTrainingDataPath());
        ChildProfile.CHILD().setTrainedDataPath(paths.getTrainedDataPath());

        log.info("Training Data Path = " + ChildProfile.CHILD().getTrainingDataPath());
        log.info("Trained Data Path  = " + ChildProfile.CHILD().getTrainedDataPath());

        // Look up the coordinator's IP from the partner list.
        IPAddress ip = this.child.getPartnerAddress(coordinatorName);
        log.info("coordinator IP = " + ip);
        if (ip != null)
        {
            ChildProfile.CHILD().setCoordinatorIP(ip);
        }
    }

    public void stop()
            throws ClassNotFoundException, IOException,
                   InterruptedException, RemoteReadException
    {
        CNNTrainer.CHILD().dispose();               // RNN -> CNN
        TerminateSignal.SIGNAL().notifyAllTermination();
        this.child.stop();
    }

    // ============================================================
    //  IDENTITY
    // ============================================================

    public String getChildKey()
    {
        return this.child.getChildKey();
    }

    // ============================================================
    //  CLUSTER OPERATIONS
    // ============================================================

    /**
     * Ask the coordinator to aggregate our gradients with the rest of the
     * cluster's, and return the resulting global norm.
     *
     * RNN -> CNN only in the type of the param list.
     */
    public double getGradientsAggregation(List<CNNParam> params) throws ClassNotFoundException, RemoteReadException, IOException
	{
		return ((GradientsAggregationResponse)this.child.read(ChildProfile.CHILD().getCoordinatorIP().getIP(), ChildProfile.CHILD().getCoordinatorIP().getPort(), new GradientsAggregationRequest(this.child.getChildKey(), params))).getNorm();
	}

    public void notifyTrainingDone() throws IOException, InterruptedException
    {
        IPAddress coord = ChildProfile.CHILD().getCoordinatorIP();
        this.child.notify(
                coord.getIP(),
                coord.getPort(),
                new TrainingDoneNotification(this.child.getChildKey()));
    }
}
