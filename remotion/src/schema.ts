export type EditScene = {
  id: string;
  from_frame: number;
  duration_frames: number;
  title: string;
  caption: string;
};

export type EditDocument = {
  schema_version: 1;
  width: number;
  height: number;
  fps: number;
  duration_frames: number;
  scenes: EditScene[];
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
};

const positiveInteger = (value: unknown, field: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${field} must be a positive integer`);
  }

  return value as number;
};

const nonNegativeInteger = (value: unknown, field: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${field} must be a non-negative integer`);
  }

  return value as number;
};

const nonEmptyString = (value: unknown, field: string): string => {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${field} must be non-empty`);
  }

  return value;
};

const parseScene = (value: unknown): EditScene => {
  if (!isRecord(value)) {
    throw new Error('scene must be an object');
  }

  return {
    id: nonEmptyString(value.id, 'scene id'),
    from_frame: nonNegativeInteger(value.from_frame, 'scene from_frame'),
    duration_frames: positiveInteger(value.duration_frames, 'scene duration_frames'),
    title: nonEmptyString(value.title, 'scene title'),
    caption: nonEmptyString(value.caption, 'scene caption'),
  };
};

export const parseEditDocument = (value: unknown): EditDocument => {
  if (!isRecord(value)) {
    throw new Error('edit document must be an object');
  }

  if (value.schema_version !== 1) {
    throw new Error('schema_version must be 1');
  }

  if (!Array.isArray(value.scenes)) {
    throw new Error('scenes must be an array');
  }

  const document: EditDocument = {
    schema_version: 1,
    width: positiveInteger(value.width, 'width'),
    height: positiveInteger(value.height, 'height'),
    fps: positiveInteger(value.fps, 'fps'),
    duration_frames: positiveInteger(value.duration_frames, 'duration_frames'),
    scenes: value.scenes.map(parseScene),
  };
  const ids = new Set<string>();

  for (const scene of document.scenes) {
    if (ids.has(scene.id)) {
      throw new Error('scene IDs must be unique');
    }
    ids.add(scene.id);

    if (scene.from_frame + scene.duration_frames > document.duration_frames) {
      throw new Error('scene exceeds composition duration');
    }
  }

  return document;
};
