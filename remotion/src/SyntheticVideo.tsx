import {
  AbsoluteFill,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import type {EditDocument, EditScene} from './schema';

const BACKGROUNDS = [
  'linear-gradient(135deg, #071a3d 0%, #000000 100%)',
  'linear-gradient(135deg, #3d071a 0%, #000000 100%)',
  'linear-gradient(135deg, #073d2a 0%, #000000 100%)',
];

const toLocalSrc = (path: string): string => {
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('file://')) {
    return path;
  }
  return staticFile(path);
};

// Deterministic pseudo-random per star index so the field is stable across
// frames within a scene but varied between scenes.
const starHash = (i: number): number => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// Animated starfield: twinkling stars with slow parallax drift. Pure CSS, no
// external assets. Adds depth behind title cards and subtle motion over media.
const StarField = ({count = 90}: {count?: number}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  return (
    <AbsoluteFill style={{pointerEvents: 'none', overflow: 'hidden'}}>
      {Array.from({length: count}, (_, i) => {
        const baseX = starHash(i) * width;
        const baseY = starHash(i + 1000) * height;
        const depth = 0.3 + starHash(i + 2000) * 0.7; // parallax factor
        const drift = frame * (4 + depth * 8);
        const twinkle =
          0.35 + 0.65 * Math.abs(Math.sin(frame * 0.04 + i));
        const size = 1 + depth * 2;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: baseX - drift,
              top: baseY,
              width: `${size}px`,
              height: `${size}px`,
              borderRadius: '50%',
              backgroundColor: '#ffffff',
              opacity: twinkle * depth,
              transform: 'translateZ(0)',
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

const SceneMedia = ({scene}: {scene: EditScene}) => {
  const frame = useCurrentFrame();
  if (scene.clip) {
    return (
      <AbsoluteFill>
        <OffthreadVideo
          loop
          muted
          src={toLocalSrc(scene.clip)}
          style={{height: '100%', objectFit: 'cover', width: '100%'}}
        />
        <AbsoluteFill style={{backgroundColor: 'rgba(0, 0, 0, 0.55)'}} />
      </AbsoluteFill>
    );
  }
  if (scene.image) {
    const zoom = interpolate(frame, [0, scene.duration_frames], [1, 1.12], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    // Slow deliberate pan so stills feel alive (Ken Burns).
    const panX = interpolate(frame, [0, scene.duration_frames], [-3, 3], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    const panY = interpolate(frame, [0, scene.duration_frames], [-2, 2], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    return (
      <AbsoluteFill>
        <Img
          src={toLocalSrc(scene.image)}
          style={{
            height: '100%',
            objectFit: 'cover',
            transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
            width: '100%',
          }}
        />
        <AbsoluteFill style={{backgroundColor: 'rgba(0, 0, 0, 0.55)'}} />
      </AbsoluteFill>
    );
  }
  return null;
};

const SceneCard = ({
  scene,
  index,
  total,
  isLast,
  sources,
  overlay,
}: {
  scene: EditScene;
  index: number;
  total: number;
  isLast: boolean;
  sources?: string[];
  overlay?: boolean;
}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const fadeIn = Math.min(15, scene.duration_frames);
  const fadeOut = Math.min(8, scene.duration_frames);

  const enter = interpolate(frame, [0, fadeIn], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const exit = interpolate(
    frame,
    [scene.duration_frames - fadeOut, scene.duration_frames],
    [1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  // During a transition overlay, ramp opacity in from zero so the next scene
  // dissolves over the previous one instead of popping in.
  const overlayIn = overlay
    ? interpolate(frame, [0, fadeIn + 6], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 1;

  const rise = interpolate(frame, [0, fadeIn], [36, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const captionIn = interpolate(frame, [fadeIn, fadeIn + 12], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background: BACKGROUNDS[index % BACKGROUNDS.length],
        opacity: overlay ? overlayIn : Math.min(enter, exit),
      }}
    >
      <SceneMedia scene={scene} />
      {/* Animated starfield for depth behind title cards */}
      <StarField count={80} />
      {/* Cinematic vignette to focus the eye and add a filmic look */}
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.6) 100%)',
          pointerEvents: 'none',
        }}
      />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'Arial, sans-serif',
          justifyContent: 'center',
          padding: '8% 10%',
          textAlign: 'center',
          transform: `translateY(${rise}px)`,
        }}
      >
        <div
          style={{
            color: '#22d3ee',
            fontSize: Math.max(20, Math.round(height * 0.032)),
            fontWeight: 700,
            letterSpacing: '0.35em',
            marginBottom: Math.round(height * 0.03),
          }}
        >
          PART {index + 1} OF {total}
          {scene.narration ? '  ·  NARRATED' : ''}
        </div>
        <div
          style={{
            color: scene.text_color ?? '#f8fafc',
            fontSize: Math.max(44, Math.round(height * 0.095)),
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.08,
            maxWidth: '85%',
            textShadow: '0 2px 12px rgba(0,0,0,0.7)',
          }}
        >
          {scene.title}
        </div>
        <div
          style={{
            backgroundColor: scene.accent_color ?? '#22d3ee',
            height: Math.max(6, Math.round(height * 0.01)),
            margin: `${Math.round(height * 0.03)}px auto 0`,
            width: '16%',
          }}
        />
        <div
          style={{
            color: '#e2e8f0',
            fontSize: Math.max(24, Math.round(height * 0.042)),
            letterSpacing: '0.02em',
            lineHeight: 1.3,
            marginTop: Math.round(height * 0.03),
            maxWidth: '80%',
            opacity: captionIn,
          }}
        >
          {scene.caption}
        </div>
        {scene.visual && (
          <div
            style={{
              color: '#64748b',
              fontSize: Math.max(15, Math.round(height * 0.024)),
              fontStyle: 'italic',
              marginTop: Math.round(height * 0.02),
              opacity: captionIn,
            }}
          >
            {scene.visual}
          </div>
        )}
      </AbsoluteFill>
      {isLast && sources && sources.length > 0 && (
        <div
          style={{
            bottom: '4%',
            color: '#475569',
            fontFamily: 'Arial, sans-serif',
            fontSize: Math.max(13, Math.round(height * 0.02)),
            left: '8%',
            opacity: captionIn,
            position: 'absolute',
            right: '8%',
            textAlign: 'center',
          }}
        >
          Sources: {sources.join('  ·  ').slice(0, 160)}
        </div>
      )}
    </AbsoluteFill>
  );
};

const ProgressBar = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        backgroundColor: 'rgba(148, 163, 184, 0.25)',
        bottom: 0,
        height: 8,
        left: 0,
        position: 'absolute',
        right: 0,
      }}
    >
      <div
        style={{
          backgroundColor: '#22d3ee',
          height: '100%',
          width: `${progress * 100}%`,
        }}
      />
    </div>
  );
};

const TRANSITION_FRAMES = 24; // ~0.8s cross-dissolve at 30fps

export const SyntheticVideo = ({scenes, sources}: EditDocument) => {
  return (
    <AbsoluteFill>
      {scenes.map((scene, index) => (
        <Sequence
          durationInFrames={scene.duration_frames}
          from={scene.from_frame}
          key={scene.id}
          layout="none"
        >
          <SceneCard
            index={index}
            isLast={index === scenes.length - 1}
            scene={scene}
            sources={sources}
            total={scenes.length}
          />
        </Sequence>
      ))}
      {/* Cross-dissolve transitions: fade the next scene in over the tail of
          the current one. Only valid within a chunk (boundaries hard-cut). */}
      {scenes.slice(0, -1).map((scene, index) => (
        <Sequence
          durationInFrames={TRANSITION_FRAMES}
          from={scene.from_frame + scene.duration_frames - TRANSITION_FRAMES}
          key={`transition-${scene.id}`}
          layout="none"
        >
          <SceneCard
            index={index + 1}
            isLast={index === scenes.length - 2}
            scene={scenes[index + 1]}
            sources={sources}
            total={scenes.length}
            overlay
          />
        </Sequence>
      ))}
      <ProgressBar />
    </AbsoluteFill>
  );
};
