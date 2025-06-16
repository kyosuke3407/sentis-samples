using UnityEngine;
using Unity.Sentis;
using System;
using System.Globalization;
using Unity.Mathematics;
using UnityEngine.UI;

public class BlazeFaceRunner : MonoBehaviour
{
    [SerializeField] private ModelAsset BlazeFaceAsset, AttentionMeshAsset;
    private Model BlazeFaceModel, AttentionMeshModel;
    private Worker BlazeFaceWorker, AttentionMeshWorker;
    private Tensor<float> inputTensor;
    private Tensor<float> attentionMeshInputTensor;
    private TextureTransform transformBF, transformAM;

    [SerializeField] private TextAsset anchorCSV;
    const int k_NumAnchors = 896;
    float[,] m_Anchors;

    public float iouThreshold = 0.3f;
    public float scoreThreshold = 0.5f;

    [SerializeField] private Texture2D tex;
    [SerializeField] private RawImage rawImage, IrisImage;
    [SerializeField] private Material material;
    RenderTexture croppedRT;
    [SerializeField] private Material irisMaterial;


    public static float[,] LoadAnchors(string csv, int k)
    {
        var anchors = new float[k, 4];
        var anchorLines = csv.Split('\n');

        for (int i = 0; i < k; i++)
        {
            var anchorValues = anchorLines[i].Split(',');
            for (var j = 0; j < 4; j++)
            {
                anchors[i, j] = float.Parse(anchorValues[j], CultureInfo.InvariantCulture);
            }
        }
        return anchors;
    }

    //Non-Maximum Suppression 
    public static (FunctionalTensor, FunctionalTensor, FunctionalTensor) NWSF(FunctionalTensor rawBoxes, FunctionalTensor rawScores, FunctionalTensor anchors, float iouTreshold, float scoreThreshold)
    {
        //uv
        var xCenter = rawBoxes[0, .., 0] + anchors[.., 0] * 128;
        var yCenter = rawBoxes[0, .., 1] + anchors[.., 1] * 128;

        var widthHalf = 0.5f * rawBoxes[0, .., 2];
        var heightHalf = 0.5f * rawBoxes[0, .., 3];

        var nmsBoxes = Functional.Stack(new[]
        {
            yCenter - heightHalf,
            xCenter - widthHalf,
            yCenter + heightHalf,
            xCenter + widthHalf,
        }, 1);

        var clamp = Functional.Sigmoid(Functional.Clamp(rawScores, -100f, 100f));
        var nmsScores = Functional.Squeeze(clamp);
        var selectedIndies = Functional.NMS(nmsBoxes, nmsScores, iouTreshold, scoreThreshold);

        var selectedBoxes = Functional.IndexSelect(rawBoxes, 1, selectedIndies).Unsqueeze(0);
        var selectedScores = Functional.IndexSelect(rawScores, 1, selectedIndies).Unsqueeze(0);

        return (selectedIndies, selectedScores, selectedBoxes);
    }

    void Start()
    {
        m_Anchors = LoadAnchors(anchorCSV.text, k_NumAnchors);

        BlazeFaceModel = ModelLoader.Load(BlazeFaceAsset);
        var graph = new FunctionalGraph();

        var input = graph.AddInput(BlazeFaceModel, 0);
        var outputs = Functional.Forward(BlazeFaceModel, 2 * input - 1);
        var boxes = outputs[0];
        var scores = outputs[1];
        var ancohorData = new float[k_NumAnchors * 4];
        Buffer.BlockCopy(m_Anchors, 0, ancohorData, 0, ancohorData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), ancohorData);
        var idxScoreBox = NWSF(boxes, scores, anchors, iouThreshold, scoreThreshold);
        var BlazeFace = graph.Compile(idxScoreBox.Item1, idxScoreBox.Item2, idxScoreBox.Item3);

        BlazeFaceWorker = new Worker(BlazeFace, BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 128, 128, 3));
        transformBF = new TextureTransform()
            .SetDimensions(128, 128, 3)
            .SetTensorLayout(TensorLayout.NHWC);


        croppedRT = new RenderTexture(192, 192, 0, RenderTextureFormat.ARGB32);
        croppedRT.filterMode = FilterMode.Bilinear;
        croppedRT.Create();


        AttentionMeshModel = ModelLoader.Load(AttentionMeshAsset);
        var AttentionMeshGraph = new FunctionalGraph();
        var inputAM = AttentionMeshGraph.AddInput(AttentionMeshModel, 0);
        var outputsAM = Functional.Forward(AttentionMeshModel, inputAM);
        var AttentionMesh = AttentionMeshGraph.Compile(outputsAM);

        AttentionMeshWorker = new Worker(AttentionMesh, BackendType.GPUCompute);
        attentionMeshInputTensor = new Tensor<float>(new TensorShape(1, 3, 192, 192));
        transformAM = new TextureTransform()
            .SetDimensions(192, 192, 3)
            .SetTensorLayout(TensorLayout.NCHW);
    }


    void Update()
    {
        rawImage.texture = tex;
        TextureConverter.ToTensor(tex, inputTensor, transformBF);
        BlazeFaceWorker.SetInput(0, inputTensor);

        BlazeFaceWorker.Schedule();

        var indexFace = BlazeFaceWorker.PeekOutput(0) as Tensor<int>;
        var score = BlazeFaceWorker.PeekOutput(1) as Tensor<float>;
        var Box = BlazeFaceWorker.PeekOutput(2) as Tensor<float>;

        var cpuIndex = indexFace.ReadbackAndClone();
        var cpuScore = score.ReadbackAndClone();
        var cpuBox = Box.ReadbackAndClone();
        var anchorPosition = 128 * new float2(m_Anchors[cpuIndex[0], 0], m_Anchors[cpuIndex[0], 1]);
        var boxSpace = anchorPosition + new float2(cpuBox[0, 0, 0], cpuBox[0, 0, 1]);
        var boxTopRightSpace = anchorPosition + new float2(cpuBox[0, 0, 0] + 0.5f * cpuBox[0, 0, 2], cpuBox[0, 0, 1] + 0.5f * cpuBox[0, 0, 3]);

        var x = boxSpace.x - cpuBox[0, 0, 2] * 0.5f;
        var y = Mathf.Abs((boxSpace.y - cpuBox[0, 0, 3] * 0.5f) - 128);

        Vector4 box = new Vector4(x, y, cpuBox[0, 0, 2], cpuBox[0, 0, 3]);
        material.SetVector("_Box", box);
        material.SetTexture("_MainTex", tex);

        Graphics.Blit(tex, croppedRT, material);
        TextureConverter.ToTensor(croppedRT, attentionMeshInputTensor, transformAM);
        AttentionMeshWorker.SetInput(0, attentionMeshInputTensor);
        AttentionMeshWorker.Schedule();

        var LeftIris = AttentionMeshWorker.PeekOutput(2) as Tensor<float>;
        var RightIris = AttentionMeshWorker.PeekOutput(6) as Tensor<float>;

        var cpuLeftIris = LeftIris.ReadbackAndClone();
        var cpuRightIris = RightIris.ReadbackAndClone();
        irisMaterial.SetTexture("_MainTex", croppedRT);
        irisMaterial.SetFloat("_LeftIrisX", cpuLeftIris[0, 0, 0, 0]);
        irisMaterial.SetFloat("_LeftIrisY", cpuLeftIris[0, 0, 0, 1]);
        irisMaterial.SetFloat("_RightIrisX", cpuRightIris[0, 0, 0, 0]);
        irisMaterial.SetFloat("_RightIrisY", cpuRightIris[0, 0, 0, 1]);
        IrisImage.texture = croppedRT;
    }
}
