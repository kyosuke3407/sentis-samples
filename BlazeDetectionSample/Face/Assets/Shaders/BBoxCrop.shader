Shader "Unlit/BBoxCrop"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white"{}
        _Box("Bounding Box (x, y, w, h)", Vector) = (0.2, 0.8, 0.4, 0.3)
        _Margin("Margin", Float) = 0.05
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _Box;
            float _Margin;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 topLeft = (_Box.xy / 128.0);
                float2 size = _Box.zw / 128.0;

                float offsetX = _Margin * size.x;
                float offsetY = _Margin * size.y;
                float2 bottomLeft = float2(topLeft.x, topLeft.y - size.y);
                float2 boxMin = bottomLeft + float2(offsetX / (-2), 0);
                float2 boxSize = float2(size.x, size.y) + float2(offsetX, offsetY);

                float2 uv = boxMin + i.uv * boxSize;

                return tex2D(_MainTex, uv);
            }
            ENDCG
        }
    }
}