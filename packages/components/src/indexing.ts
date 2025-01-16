import { VectorStore } from '@langchain/core/vectorstores'
import { v5 as uuidv5 } from 'uuid'
import { RecordManagerInterface, UUIDV5_NAMESPACE } from '@langchain/community/indexes/base'
import { insecureHash } from '@langchain/core/utils/hash'
import { Document, DocumentInterface } from '@langchain/core/documents'
import { BaseDocumentLoader } from 'langchain/document_loaders/base.js'
import { IndexingResult } from './Interface'

type StringOrDocFunc = string | ((doc: DocumentInterface) => string)

export interface HashedDocumentInterface extends DocumentInterface {
    uid: string
    hash_: string
    contentHash?: string
    metadataHash?: string
    calculateHashes(): void
    toDocument(): DocumentInterface
    isHashUid(): boolean
}

/**
 * HashedDocument is a Document with hashes calculated.
 * Hashes are calculated based on page content and metadata.
 * It is used for indexing.
 */
export class _HashedDocument extends Document implements HashedDocumentInterface {
    private __hash_?: HashedDocumentInterface['hash_']

    contentHash?: string

    metadataHash?: string

    get uid() {
        return this.id || this.hash_
    }

    get hash_() {
        if (!this.__hash_) {
            this.calculateHashes()
        }

        return this.__hash_!
    }

    isHashUid() {
        return !this.id
    }

    calculateHashes(): void {
        const forbiddenKeys = ['hash_', 'content_hash', 'metadata_hash']

        for (const key of forbiddenKeys) {
            if (key in this.metadata) {
                throw new Error(
                    `Metadata cannot contain key ${key} as it is reserved for internal use. Restricted keys: [${forbiddenKeys.join(', ')}]`
                )
            }
        }

        const contentHash = this._hashStringToUUID(this.pageContent)

        try {
            const metadataHash = this._hashNestedDictToUUID(this.metadata)
            this.contentHash = contentHash
            this.metadataHash = metadataHash
        } catch (e) {
            throw new Error(`Failed to hash metadata: ${e}. Please use a dict that can be serialized using json.`)
        }

        this.__hash_ = this._hashStringToUUID(this.contentHash + this.metadataHash)
    }

    toDocument(): DocumentInterface {
        return new Document({
            pageContent: this.pageContent,
            metadata: this.metadata
        })
    }

    static fromDocument(document: DocumentInterface): _HashedDocument {
        const doc = new this(document)
        doc.calculateHashes()
        return doc
    }

    private _hashStringToUUID(inputString: string): string {
        const hash_value = insecureHash(inputString)
        return uuidv5(hash_value, UUIDV5_NAMESPACE)
    }

    private _hashNestedDictToUUID(data: Record<string, unknown>): string {
        const serialized_data = JSON.stringify(data, Object.keys(data).sort())
        const hash_value = insecureHash(serialized_data)
        return uuidv5(hash_value, UUIDV5_NAMESPACE)
    }
}

export type CleanupMode = 'full' | 'incremental'

export type IndexOptions = {
    /**
     * The number of documents to index in one batch.
     */
    batchSize?: number
    /**
     * The cleanup mode to use. Can be "full", "incremental" or undefined.
     * - **Incremental**: Cleans up all documents that haven't been updated AND
     *   that are associated with source ids that were seen
     *   during indexing.
     *   Clean up is done continuously during indexing helping
     *   to minimize the probability of users seeing duplicated
     *   content.
     * - **Full**: Delete all documents that haven to been returned by the loader.
     *   Clean up runs after all documents have been indexed.
     *   This means that users may see duplicated content during indexing.
     * - **undefined**: Do not delete any documents.
     */
    cleanup?: CleanupMode
    /**
     * Optional key that helps identify the original source of the document.
     * Must either be a string representing the key of the source in the metadata
     * or a function that takes a document and returns a string representing the source.
     * **Required when cleanup is incremental**.
     */
    sourceIdKey?: StringOrDocFunc
    /**
     * Batch size to use when cleaning up documents.
     */
    cleanupBatchSize?: number
    /**
     * Force update documents even if they are present in the
     * record manager. Useful if you are re-indexing with updated embeddings.
     */
    forceUpdate?: boolean

    vectorStoreName?: string
}

export function _batch<T>(size: number, iterable: T[]): T[][] {
    const batches: T[][] = []
    let currentBatch: T[] = []

    iterable.forEach((item) => {
        currentBatch.push(item)

        if (currentBatch.length >= size) {
            batches.push(currentBatch)
            currentBatch = []
        }
    })

    if (currentBatch.length > 0) {
        batches.push(currentBatch)
    }

    return batches
}

export function _deduplicateInOrder(hashedDocuments: HashedDocumentInterface[]): HashedDocumentInterface[] {
    const seen = new Set<string>()
    const deduplicated: HashedDocumentInterface[] = []

    for (const hashedDoc of hashedDocuments) {
        if (!seen.has(hashedDoc.hash_)) {
            seen.add(hashedDoc.hash_)
            deduplicated.push(hashedDoc)
        }
    }
    return deduplicated
}

export function _getSourceIdAssigner(sourceIdKey: StringOrDocFunc | null): (doc: DocumentInterface) => string | null {
    if (sourceIdKey === null) {
        return (_doc: DocumentInterface) => null
    } else if (typeof sourceIdKey === 'string') {
        return (doc: DocumentInterface) => doc.metadata[sourceIdKey]
    } else if (typeof sourceIdKey === 'function') {
        return sourceIdKey
    } else {
        throw new Error(`sourceIdKey should be null, a string or a function, got ${typeof sourceIdKey}`)
    }
}

export const _isBaseDocumentLoader = (arg: any): arg is BaseDocumentLoader => {
    if ('load' in arg && typeof arg.load === 'function' && 'loadAndSplit' in arg && typeof arg.loadAndSplit === 'function') {
        return true
    }
    return false
}

interface IndexArgs {
    docsSource: BaseDocumentLoader | DocumentInterface[]
    recordManager: RecordManagerInterface
    vectorStore: VectorStore
    options?: IndexOptions
}

/**
 * Index data from the doc source into the vector store.
 *
 * Indexing functionality uses a manager to keep track of which documents
 * are in the vector store.
 *
 * This allows us to keep track of which documents were updated, and which
 * documents were deleted, which documents should be skipped.
 *
 * For the time being, documents are indexed using their hashes, and users
 *  are not able to specify the uid of the document.
 *
 * @param {IndexArgs} args
 * @param {BaseDocumentLoader | DocumentInterface[]} args.docsSource The source of documents to index. Can be a DocumentLoader or a list of Documents.
 * @param {RecordManagerInterface} args.recordManager The record manager to use for keeping track of indexed documents.
 * @param {VectorStore} args.vectorStore The vector store to use for storing the documents.
 * @param {IndexOptions | undefined} args.options Options for indexing.
 * @returns {Promise<IndexingResult>}
 */
export async function index(args: IndexArgs): Promise<IndexingResult> {
    const { docsSource, recordManager, vectorStore, options } = args
    const { batchSize = 100, cleanup, sourceIdKey, cleanupBatchSize = 1000, forceUpdate = false, vectorStoreName } = options ?? {}

    if (cleanup === 'incremental' && !sourceIdKey) {
        throw new Error("sourceIdKey is required when cleanup mode is incremental. Please provide through 'options.sourceIdKey'.")
    }

    if (vectorStoreName) {
        ;(recordManager as any).namespace = (recordManager as any).namespace + '_' + vectorStoreName
    }

    const docs = _isBaseDocumentLoader(docsSource) ? await docsSource.load() : docsSource

    const sourceIdAssigner = _getSourceIdAssigner(sourceIdKey ?? null)

    const indexStartDt = await recordManager.getTime()
    let numAdded = 0
    let addedDocs: Document[] = []
    let numDeleted = 0
    let numUpdated = 0
    let numSkipped = 0
    let totalKeys = 0

    const batches = _batch<DocumentInterface>(batchSize ?? 100, docs)

    for (const batch of batches) {
        const hashedDocs = _deduplicateInOrder(batch.map((doc) => _HashedDocument.fromDocument(doc)))

        const sourceIds = hashedDocs.map((doc) => sourceIdAssigner(doc))

        if (cleanup === 'incremental') {
            hashedDocs.forEach((_hashedDoc, index) => {
                const source = sourceIds[index]
                if (source === null) {
                    throw new Error('sourceIdKey must be provided when cleanup is incremental')
                }
            })
        }

        const [batchExistsByUid, batchExistsByHash] = await Promise.all([
            recordManager.exists(hashedDocs.map((doc) => doc.uid)),
            recordManager.exists(hashedDocs.map((doc) => doc.hash_))
        ])

        const uids: string[] = []
        const docsToIndex: DocumentInterface[] = []
        const docsToUpdate: string[] = []
        const seenDocs = new Set<string>()

        /**
         * TODO: Find a way to check if document is really the right document or a collision due to hash id
         **/
        hashedDocs.forEach((hashedDoc, i) => {
            const uidExists = batchExistsByUid[i]
            const hashExists = batchExistsByHash[i]
            let existingKey: string | undefined

            // Is matched by UID only
            if (uidExists && !hashExists) {
                existingKey = hashedDoc.uid
            }
            // Is matched by Hash only
            else if (!uidExists && hashExists) {
                existingKey = hashedDoc.hash_
            }
            // Is matched by UID & Hash
            else if (uidExists && hashExists) {
                // Hash is not the UID, there is probably a collision of UID against a legacy Hash UID
                if (!hashedDoc.isHashUid()) {
                    throw Error(`Document UID collision detected with UID ${hashedDoc.uid}`)
                }

                // Hash is used as UID, that could be ok in some case (not using DocumentStore), but risky because of collisions
                existingKey = hashedDoc.uid
            }

            if (existingKey) {
                if (forceUpdate) {
                    seenDocs.add(existingKey)
                } else {
                    docsToUpdate.push(existingKey)
                    return
                }
            }

            uids.push(existingKey || hashedDoc.uid)
            docsToIndex.push(hashedDoc.toDocument())
        })

        if (docsToUpdate.length > 0) {
            await recordManager.update(docsToUpdate, { timeAtLeast: indexStartDt })
            numSkipped += docsToUpdate.length
        }

        if (docsToIndex.length > 0) {
            await vectorStore.addDocuments(docsToIndex, { ids: uids })
            const newDocs = docsToIndex.map((docs) => ({
                pageContent: docs.pageContent,
                metadata: docs.metadata
            }))
            addedDocs.push(...newDocs)
            numAdded += docsToIndex.length - seenDocs.size
            numUpdated += seenDocs.size
        }

        await recordManager.update(
            hashedDocs.map((doc) => doc.uid),
            { timeAtLeast: indexStartDt, groupIds: sourceIds }
        )

        if (cleanup === 'incremental') {
            sourceIds.forEach((sourceId) => {
                if (!sourceId) throw new Error('Source id cannot be null')
            })
            const uidsToDelete = await recordManager.listKeys({
                before: indexStartDt,
                groupIds: sourceIds
            })
            await vectorStore.delete({ ids: uidsToDelete })
            await recordManager.deleteKeys(uidsToDelete)
            numDeleted += uidsToDelete.length
        }
    }

    if (cleanup === 'full') {
        let uidsToDelete = await recordManager.listKeys({
            before: indexStartDt,
            limit: cleanupBatchSize
        })
        while (uidsToDelete.length > 0) {
            await vectorStore.delete({ ids: uidsToDelete })
            await recordManager.deleteKeys(uidsToDelete)
            numDeleted += uidsToDelete.length
            uidsToDelete = await recordManager.listKeys({
                before: indexStartDt,
                limit: cleanupBatchSize
            })
        }
    }

    totalKeys = (await recordManager.listKeys({})).length

    return {
        numAdded,
        numDeleted,
        numUpdated,
        numSkipped,
        totalKeys,
        addedDocs
    }
}
